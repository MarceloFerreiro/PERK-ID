import 'dart:convert';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

List<CameraDescription> _cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  _cameras = await availableCameras();
  runApp(const PerkIDApp());
}

class PerkIDApp extends StatelessWidget {
  const PerkIDApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PerkID',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: Colors.indigo,
          brightness: Brightness.dark,
        ),
        useMaterial3: true,
      ),
      home: const CameraPage(),
    );
  }
}

// ── Models ─────────────────────────────────────────────────────────────────────

class Substance {
  final String nombre;
  final double? valor;
  final String? unidad;
  Substance({required this.nombre, this.valor, this.unidad});
  factory Substance.fromJson(Map<String, dynamic> j) => Substance(
        nombre: j['nombre'] as String,
        valor: (j['valor'] as num?)?.toDouble(),
        unidad: j['unidad'] as String?,
      );
}

class PillResult {
  final int rank;
  final double distance;
  final String imageUrl;
  final String? fecha;
  final String? logo;
  final String? color;
  final String? procedencia;
  final double? pesoMg;
  final double? diametroMm;
  final bool? divisible;
  final List<Substance> substances;

  PillResult({
    required this.rank,
    required this.distance,
    required this.imageUrl,
    this.fecha,
    this.logo,
    this.color,
    this.procedencia,
    this.pesoMg,
    this.diametroMm,
    this.divisible,
    required this.substances,
  });

  factory PillResult.fromJson(Map<String, dynamic> j, String baseUrl) =>
      PillResult(
        rank: j['rank'] as int,
        distance: (j['distance'] as num).toDouble(),
        imageUrl: '$baseUrl${j['image_url']}',
        fecha: j['fecha'] as String?,
        logo: j['logo'] as String?,
        color: j['color'] as String?,
        procedencia: j['procedencia'] as String?,
        pesoMg: (j['peso_mg'] as num?)?.toDouble(),
        diametroMm: (j['diametro_mm'] as num?)?.toDouble(),
        divisible: j['divisible'] as bool?,
        substances: (j['substances'] as List)
            .map((s) => Substance.fromJson(s as Map<String, dynamic>))
            .toList(),
      );
}

// ── Camera Page ────────────────────────────────────────────────────────────────

class CameraPage extends StatefulWidget {
  const CameraPage({super.key});

  @override
  State<CameraPage> createState() => _CameraPageState();
}

class _CameraPageState extends State<CameraPage> with WidgetsBindingObserver {
  CameraController? _controller;
  final _serverCtrl = TextEditingController();
  bool _searching = false;
  String? _error;
  int _topk = 10;
  FlashMode _flashMode = FlashMode.off;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadPrefs().then((_) => _initCamera());
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      ctrl.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  Future<void> _loadPrefs() async {
    final prefs = await SharedPreferences.getInstance();
    _serverCtrl.text =
        prefs.getString('server_url') ?? 'http://192.168.1.137:8000';
  }

  Future<void> _savePrefs() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('server_url', _serverCtrl.text.trim());
  }

  Future<void> _initCamera() async {
    if (_cameras.isEmpty) return;
    final back = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => _cameras.first,
    );
    final ctrl = CameraController(
      back,
      ResolutionPreset.high,
      enableAudio: false,
    );
    await ctrl.initialize();
    if (!mounted) return;
    setState(() => _controller = ctrl);
  }

  Future<void> _toggleFlash() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized) return;
    final next = _flashMode == FlashMode.off ? FlashMode.torch : FlashMode.off;
    await ctrl.setFlashMode(next);
    setState(() => _flashMode = next);
  }

  String get _baseUrl {
    var url = _serverCtrl.text.trim();
    if (url.endsWith('/')) url = url.substring(0, url.length - 1);
    return url;
  }

  Future<void> _capture() async {
    final ctrl = _controller;
    if (ctrl == null || !ctrl.value.isInitialized || _searching) return;

    setState(() {
      _searching = true;
      _error = null;
    });

    try {
      final file = await ctrl.takePicture();
      await _sendImage(file);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _searching = false);
    }
  }

  Future<void> _pickFromGallery() async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.gallery, imageQuality: 85);
    if (picked == null) return;

    setState(() {
      _searching = true;
      _error = null;
    });
    try {
      await _sendImage(picked);
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _searching = false);
    }
  }

  Future<void> _sendImage(XFile file) async {
    await _savePrefs();
    final bytes = await file.readAsBytes();
    final uri = Uri.parse('$_baseUrl/query?topk=$_topk');
    final req = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes(
        'image',
        bytes,
        filename: file.name.isNotEmpty ? file.name : 'query.jpg',
      ));

    final streamed = await req.send().timeout(const Duration(seconds: 60));
    final body = await streamed.stream.bytesToString();

    if (streamed.statusCode != 200) {
      final detail = (jsonDecode(body) as Map)['detail'] ?? 'Error ${streamed.statusCode}';
      throw Exception(detail);
    }

    final results = (jsonDecode(body)['results'] as List)
        .map((r) => PillResult.fromJson(r as Map<String, dynamic>, _baseUrl))
        .toList();

    if (!mounted) return;
    await showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _ResultsSheet(results: results),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // ── Camera preview ──────────────────────────────────────
          if (_controller != null && _controller!.value.isInitialized)
            CameraPreview(_controller!)
          else
            const Center(child: CircularProgressIndicator()),

          // ── Corner gradient for controls ────────────────────────
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: Container(
              height: 160,
              decoration: const BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.bottomCenter,
                  end: Alignment.topCenter,
                  colors: [Colors.black87, Colors.transparent],
                ),
              ),
            ),
          ),

          // ── Error banner ────────────────────────────────────────
          if (_error != null)
            Positioned(
              top: MediaQuery.of(context).padding.top + 8,
              left: 16, right: 16,
              child: Material(
                color: Colors.red.shade800,
                borderRadius: BorderRadius.circular(8),
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(_error!, style: const TextStyle(color: Colors.white)),
                ),
              ),
            ),

          // ── Searching overlay ───────────────────────────────────
          if (_searching)
            Container(
              color: Colors.black54,
              child: const Center(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 16),
                    Text('Searching…', style: TextStyle(color: Colors.white, fontSize: 18)),
                  ],
                ),
              ),
            ),

          // ── Bottom controls ─────────────────────────────────────
          Positioned(
            bottom: MediaQuery.of(context).padding.bottom + 24,
            left: 0, right: 0,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Gallery
                IconButton(
                  icon: const Icon(Icons.photo_library, color: Colors.white, size: 32),
                  onPressed: _searching ? null : _pickFromGallery,
                ),
                // Shutter
                GestureDetector(
                  onTap: _capture,
                  child: Container(
                    width: 72, height: 72,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      border: Border.all(color: Colors.white, width: 4),
                      color: _searching ? Colors.white24 : Colors.white30,
                    ),
                    child: const SizedBox.shrink(),
                  ),
                ),
                // Settings
                IconButton(
                  icon: const Icon(Icons.settings, color: Colors.white, size: 32),
                  onPressed: _showSettings,
                ),
              ],
            ),
          ),

          // ── Top-left: flash toggle ──────────────────────────────
          Positioned(
            top: MediaQuery.of(context).padding.top + 8,
            left: 12,
            child: GestureDetector(
              onTap: _toggleFlash,
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: _flashMode == FlashMode.torch
                      ? Colors.amber.withValues(alpha: 0.85)
                      : Colors.black54,
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _flashMode == FlashMode.torch ? Icons.flash_on : Icons.flash_off,
                  color: Colors.white,
                  size: 26,
                ),
              ),
            ),
          ),

          // ── Top-right: topK selector ────────────────────────────
          Positioned(
            top: MediaQuery.of(context).padding.top + 8,
            right: 12,
            child: _TopKSelector(
              value: _topk,
              onChanged: (v) => setState(() => _topk = v),
            ),
          ),
        ],
      ),
    );
  }

  void _showSettings() {
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Server URL'),
        content: TextField(
          controller: _serverCtrl,
          decoration: const InputDecoration(
            hintText: 'http://192.168.x.x:8000',
            border: OutlineInputBorder(),
          ),
          keyboardType: TextInputType.url,
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text('Cancel')),
          FilledButton(
            onPressed: () { _savePrefs(); Navigator.pop(ctx); },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }
}

// ── Top-K selector chip ────────────────────────────────────────────────────────

class _TopKSelector extends StatelessWidget {
  final int value;
  final ValueChanged<int> onChanged;
  const _TopKSelector({required this.value, required this.onChanged});

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<int>(
      initialValue: value,
      onSelected: onChanged,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black54,
          borderRadius: BorderRadius.circular(20),
        ),
        child: Text('Top $value', style: const TextStyle(color: Colors.white)),
      ),
      itemBuilder: (_) => [5, 10, 20]
          .map((k) => PopupMenuItem(value: k, child: Text('Top $k')))
          .toList(),
    );
  }
}

// ── Results Bottom Sheet ───────────────────────────────────────────────────────

class _ResultsSheet extends StatelessWidget {
  final List<PillResult> results;
  const _ResultsSheet({required this.results});

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      initialChildSize: 0.55,
      minChildSize: 0.3,
      maxChildSize: 0.95,
      builder: (_, scrollCtrl) => Container(
        decoration: const BoxDecoration(
          color: Color(0xFF1E1E2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(
          children: [
            // Drag handle
            Container(
              margin: const EdgeInsets.symmetric(vertical: 10),
              width: 40, height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Row(
                children: [
                  Text('${results.length} results',
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white)),
                  const Spacer(),
                  IconButton(
                    icon: const Icon(Icons.close, color: Colors.white54),
                    onPressed: () => Navigator.pop(context),
                  ),
                ],
              ),
            ),
            const Divider(height: 1, color: Colors.white12),
            Expanded(
              child: ListView.builder(
                controller: scrollCtrl,
                itemCount: results.length,
                itemBuilder: (_, i) => _PillCard(result: results[i]),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Pill Card ──────────────────────────────────────────────────────────────────

class _PillCard extends StatelessWidget {
  final PillResult result;
  const _PillCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => PillDetailPage(result: result)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
        child: Row(
          children: [
            // Rank
            SizedBox(
              width: 28,
              child: Text(
                '${result.rank}',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  color: result.rank == 1 ? Colors.amber : Colors.white38,
                ),
              ),
            ),
            const SizedBox(width: 10),
            // Image
            ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: Image.network(
                result.imageUrl,
                width: 64, height: 64,
                fit: BoxFit.cover,
                errorBuilder: (_, __, ___) =>
                    const Icon(Icons.broken_image, size: 64, color: Colors.white24),
              ),
            ),
            const SizedBox(width: 12),
            // Info
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (result.logo != null)
                    Text(result.logo!,
                        style: const TextStyle(
                            fontWeight: FontWeight.bold, color: Colors.white)),
                  if (result.color != null)
                    Text(result.color!,
                        style: const TextStyle(color: Colors.white60, fontSize: 13)),
                  if (result.substances.isNotEmpty)
                    Text(
                      result.substances.map((s) => s.nombre).join(', '),
                      style: const TextStyle(color: Colors.white60, fontSize: 12),
                      overflow: TextOverflow.ellipsis,
                    ),
                ],
              ),
            ),
            // Distance badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
              decoration: BoxDecoration(
                color: Colors.white10,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Text(
                result.distance.toStringAsFixed(3),
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Pill Detail Page ───────────────────────────────────────────────────────────

class PillDetailPage extends StatelessWidget {
  final PillResult result;
  const PillDetailPage({super.key, required this.result});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E2E),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1E1E2E),
        title: Text(result.logo ?? 'Pill #${result.rank}',
            style: const TextStyle(color: Colors.white)),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.network(
                  result.imageUrl,
                  width: 220, height: 220,
                  fit: BoxFit.contain,
                  errorBuilder: (_, __, ___) =>
                      const Icon(Icons.broken_image, size: 120, color: Colors.white24),
                ),
              ),
            ),
            const SizedBox(height: 24),
            _Section(title: 'Identity', rows: [
              _RowData('Logo', result.logo),
              _RowData('Color', result.color),
              _RowData('Origin', result.procedencia),
              _RowData('Date', result.fecha),
              _RowData('Match score', result.distance.toStringAsFixed(6)),
            ]),
            _Section(title: 'Physical', rows: [
              _RowData('Weight', result.pesoMg != null ? '${result.pesoMg} mg' : null),
              _RowData('Diameter', result.diametroMm != null ? '${result.diametroMm} mm' : null),
              _RowData('Divisible', result.divisible != null ? (result.divisible! ? 'Yes' : 'No') : null),
            ]),
            if (result.substances.isNotEmpty)
              _Section(
                title: 'Substances',
                rows: result.substances
                    .map((s) => _RowData(
                          s.nombre,
                          s.valor != null
                              ? '${s.valor} ${s.unidad ?? ''}'.trim()
                              : s.unidad,
                        ))
                    .toList(),
              ),
          ],
        ),
      ),
    );
  }
}

class _RowData {
  final String label;
  final String? value;
  const _RowData(this.label, this.value);
}

class _Section extends StatelessWidget {
  final String title;
  final List<_RowData> rows;
  const _Section({required this.title, required this.rows});

  @override
  Widget build(BuildContext context) {
    final visible = rows.where((r) => r.value != null && r.value!.isNotEmpty).toList();
    if (visible.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(title,
            style: const TextStyle(
                color: Colors.white70, fontSize: 13, letterSpacing: 1.2)),
        const SizedBox(height: 8),
        ...visible.map((r) => Padding(
              padding: const EdgeInsets.symmetric(vertical: 4),
              child: Row(
                children: [
                  SizedBox(
                    width: 100,
                    child: Text(r.label,
                        style: const TextStyle(color: Colors.white38, fontSize: 14)),
                  ),
                  Expanded(
                    child: Text(r.value!,
                        style: const TextStyle(color: Colors.white, fontSize: 14)),
                  ),
                ],
              ),
            )),
        const SizedBox(height: 20),
      ],
    );
  }
}
