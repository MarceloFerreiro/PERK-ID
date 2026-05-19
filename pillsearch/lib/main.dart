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
  List<String> _availableColors = [];
  List<String> _availableLogos  = [];
  Set<String>  _selectedColors  = {};
  Set<String>  _selectedLogos   = {};

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _loadPrefs().then((_) {
      _initCamera();
      _fetchFilters();
    });
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

  Future<void> _fetchFilters() async {
    try {
      final res = await http
          .get(Uri.parse('$_baseUrl/filters'))
          .timeout(const Duration(seconds: 10));
      if (res.statusCode == 200) {
        final data = jsonDecode(res.body) as Map<String, dynamic>;
        setState(() {
          _availableColors = List<String>.from(data['colors'] as List);
          _availableLogos  = List<String>.from(data['logos']  as List);
        });
      }
    } catch (_) {}
  }

  void _openFilterSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (_) => _FilterSheet(
        availableColors: _availableColors,
        availableLogos:  _availableLogos,
        selectedColors:  Set.from(_selectedColors),
        selectedLogos:   Set.from(_selectedLogos),
        onApply: (colors, logos) => setState(() {
          _selectedColors = colors;
          _selectedLogos  = logos;
        }),
      ),
    );
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
    var rawUrl = '$_baseUrl/query?topk=$_topk';
    for (final c in _selectedColors) rawUrl += '&colors=${Uri.encodeComponent(c)}';
    for (final l in _selectedLogos)  rawUrl += '&logos=${Uri.encodeComponent(l)}';
    final uri = Uri.parse(rawUrl);
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

          // ── Top-right: topK selector + filter button ───────────
          Positioned(
            top: MediaQuery.of(context).padding.top + 8,
            right: 12,
            child: Row(
              children: [
                _TopKSelector(value: _topk, onChanged: (v) => setState(() => _topk = v)),
                const SizedBox(width: 8),
                GestureDetector(
                  onTap: _openFilterSheet,
                  child: Container(
                    padding: const EdgeInsets.all(8),
                    decoration: BoxDecoration(
                      color: (_selectedColors.isNotEmpty || _selectedLogos.isNotEmpty)
                          ? Colors.indigo.withValues(alpha: 0.9)
                          : Colors.black54,
                      shape: BoxShape.circle,
                    ),
                    child: Stack(
                      clipBehavior: Clip.none,
                      children: [
                        const Icon(Icons.filter_list, color: Colors.white, size: 22),
                        if (_selectedColors.isNotEmpty || _selectedLogos.isNotEmpty)
                          Positioned(
                            top: -4, right: -4,
                            child: Container(
                              width: 14, height: 14,
                              decoration: const BoxDecoration(
                                color: Colors.amber,
                                shape: BoxShape.circle,
                              ),
                              child: Center(
                                child: Text(
                                  '${_selectedColors.length + _selectedLogos.length}',
                                  style: const TextStyle(fontSize: 9, fontWeight: FontWeight.bold),
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),

          // ── Active filter chips ─────────────────────────────────
          if (_selectedColors.isNotEmpty || _selectedLogos.isNotEmpty)
            Positioned(
              bottom: MediaQuery.of(context).padding.bottom + 24 + 88,
              left: 8, right: 8,
              child: SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                child: Row(
                  children: [
                    ..._selectedColors.map((c) => _ActiveFilterChip(
                      label: c,
                      color: Colors.indigo.shade700,
                      onRemove: () => setState(() => _selectedColors.remove(c)),
                    )),
                    ..._selectedLogos.map((l) => _ActiveFilterChip(
                      label: l,
                      color: Colors.deepPurple.shade700,
                      onRemove: () => setState(() => _selectedLogos.remove(l)),
                    )),
                  ],
                ),
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
            if (result.substances.isNotEmpty) ...[
              _SubstanceBar(substances: result.substances, pesoMg: result.pesoMg),
              const SizedBox(height: 8),
              _Section(
                title: 'Substances',
                rows: result.substances
                    .map((s) => _RowData(
                          s.nombre,
                          s.valor != null
                              ? '${s.valor} ${s.unidad ?? ''}'.trim()
                              : '—',
                        ))
                    .toList(),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

// ── Active filter chip (camera overlay) ───────────────────────────────────────

class _ActiveFilterChip extends StatelessWidget {
  final String label;
  final Color color;
  final VoidCallback onRemove;
  const _ActiveFilterChip({required this.label, required this.color, required this.onRemove});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(right: 6),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: const TextStyle(color: Colors.white, fontSize: 12)),
          const SizedBox(width: 4),
          GestureDetector(
            onTap: onRemove,
            child: const Icon(Icons.close, size: 14, color: Colors.white70),
          ),
        ],
      ),
    );
  }
}

// ── Filter Sheet ───────────────────────────────────────────────────────────────

Widget _colorDot(String name) {
  const map = {
    'verde': Color(0xFF4CAF50),    'azul': Color(0xFF2196F3),
    'rojo': Color(0xFFF44336),     'amarillo': Color(0xFFFFEB3B),
    'blanco': Colors.white,        'negro': Color(0xFF212121),
    'naranja': Color(0xFFFF9800),  'rosa': Color(0xFFE91E63),
    'morado': Color(0xFF9C27B0),   'lila': Color(0xFFCE93D8),
    'gris': Color(0xFF9E9E9E),     'marrón': Color(0xFF795548),
    'turquesa': Color(0xFF00BCD4),
  };
  return Container(
    width: 13, height: 13,
    decoration: BoxDecoration(
      color: map[name.toLowerCase()] ?? const Color(0xFF555555),
      shape: BoxShape.circle,
      border: Border.all(color: Colors.white24, width: 0.5),
    ),
  );
}

Widget _sectionHeader(String title, int selected) => Padding(
  padding: const EdgeInsets.fromLTRB(16, 16, 16, 4),
  child: Row(children: [
    Text(title.toUpperCase(),
        style: const TextStyle(
            color: Colors.white54, fontSize: 11,
            letterSpacing: 1.2, fontWeight: FontWeight.bold)),
    if (selected > 0) ...[
      const SizedBox(width: 8),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
        decoration: BoxDecoration(
            color: Colors.indigo, borderRadius: BorderRadius.circular(10)),
        child: Text('$selected',
            style: const TextStyle(color: Colors.white, fontSize: 10)),
      ),
    ],
  ]),
);

class _FilterSheet extends StatefulWidget {
  final List<String> availableColors;
  final List<String> availableLogos;
  final Set<String> selectedColors;
  final Set<String> selectedLogos;
  final void Function(Set<String> colors, Set<String> logos) onApply;

  const _FilterSheet({
    required this.availableColors,
    required this.availableLogos,
    required this.selectedColors,
    required this.selectedLogos,
    required this.onApply,
  });

  @override
  State<_FilterSheet> createState() => _FilterSheetState();
}

class _FilterSheetState extends State<_FilterSheet> {
  late final TextEditingController _search;
  late Set<String> _selColors;
  late Set<String> _selLogos;

  @override
  void initState() {
    super.initState();
    _search = TextEditingController();
    _selColors = Set.from(widget.selectedColors);
    _selLogos  = Set.from(widget.selectedLogos);
  }

  @override
  void dispose() { _search.dispose(); super.dispose(); }

  List<String> _filter(List<String> list) {
    final q = _search.text.toLowerCase();
    return q.isEmpty ? list : list.where((s) => s.toLowerCase().contains(q)).toList();
  }

  @override
  Widget build(BuildContext context) {
    final filteredColors = _filter(widget.availableColors);
    final filteredLogos  = _filter(widget.availableLogos);
    final totalSelected  = _selColors.length + _selLogos.length;

    return DraggableScrollableSheet(
      initialChildSize: 0.72,
      minChildSize: 0.4,
      maxChildSize: 0.95,
      builder: (_, ctrl) => Container(
        decoration: const BoxDecoration(
          color: Color(0xFF1E1E2E),
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: Column(children: [
          Container(
            margin: const EdgeInsets.symmetric(vertical: 10),
            width: 40, height: 4,
            decoration: BoxDecoration(
                color: Colors.white24, borderRadius: BorderRadius.circular(2)),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
            child: TextField(
              controller: _search,
              onChanged: (_) => setState(() {}),
              autofocus: true,
              style: const TextStyle(color: Colors.white),
              decoration: InputDecoration(
                hintText: 'Buscar color o logo…',
                hintStyle: const TextStyle(color: Colors.white38),
                prefixIcon: const Icon(Icons.search, color: Colors.white38),
                filled: true,
                fillColor: Colors.white10,
                border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                    borderSide: BorderSide.none),
                suffixIcon: _search.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear, color: Colors.white38),
                        onPressed: () { _search.clear(); setState(() {}); })
                    : null,
              ),
            ),
          ),
          const Divider(height: 1, color: Colors.white12),
          Expanded(
            child: ListView(controller: ctrl, children: [
              if (filteredColors.isNotEmpty) ...[
                _sectionHeader('Colores', _selColors.length),
                ...filteredColors.map((c) => CheckboxListTile(
                  dense: true,
                  value: _selColors.contains(c),
                  onChanged: (v) => setState(
                      () => v! ? _selColors.add(c) : _selColors.remove(c)),
                  title: Row(children: [
                    _colorDot(c),
                    const SizedBox(width: 10),
                    Text(c, style: const TextStyle(color: Colors.white)),
                  ]),
                  activeColor: Colors.indigo,
                  checkColor: Colors.white,
                  side: const BorderSide(color: Colors.white24),
                )),
              ],
              if (filteredLogos.isNotEmpty) ...[
                _sectionHeader('Logos', _selLogos.length),
                ...filteredLogos.map((l) => CheckboxListTile(
                  dense: true,
                  value: _selLogos.contains(l),
                  onChanged: (v) => setState(
                      () => v! ? _selLogos.add(l) : _selLogos.remove(l)),
                  title: Text(l, style: const TextStyle(color: Colors.white)),
                  activeColor: Colors.indigo,
                  checkColor: Colors.white,
                  side: const BorderSide(color: Colors.white24),
                )),
              ],
              if (filteredColors.isEmpty && filteredLogos.isEmpty)
                const Padding(
                  padding: EdgeInsets.all(32),
                  child: Center(
                    child: Text('Sin resultados',
                        style: TextStyle(color: Colors.white38)),
                  ),
                ),
              const SizedBox(height: 80),
            ]),
          ),
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
              child: Row(children: [
                if (totalSelected > 0)
                  TextButton(
                    onPressed: () => setState(() {
                      _selColors.clear(); _selLogos.clear();
                    }),
                    child: const Text('Limpiar',
                        style: TextStyle(color: Colors.white54)),
                  ),
                const Spacer(),
                FilledButton(
                  onPressed: () {
                    widget.onApply(_selColors, _selLogos);
                    Navigator.pop(context);
                  },
                  child: Text(totalSelected > 0
                      ? 'Aplicar ($totalSelected)'
                      : 'Aplicar'),
                ),
              ]),
            ),
          ),
        ]),
      ),
    );
  }
}

// ── Substance composition bar ──────────────────────────────────────────────────

class _SubstanceBar extends StatelessWidget {
  final List<Substance> substances;
  final double? pesoMg;
  const _SubstanceBar({required this.substances, this.pesoMg});

  static const _palette = [
    Color(0xFF4C9BE8),
    Color(0xFFE8994C),
    Color(0xFF4CE884),
    Color(0xFFE84C4C),
    Color(0xFFB44CE8),
    Color(0xFFE8E84C),
    Color(0xFF4CE8E8),
    Color(0xFFE84CB4),
  ];

  static const _grey     = Color(0xFF555566);
  static const _greyText = TextStyle(color: Colors.white38, fontSize: 12);
  static const _labelStyle = TextStyle(color: Colors.white70, fontSize: 13, letterSpacing: 1.2);

  Widget _dot(Color color) => Container(
        width: 10, height: 10,
        decoration: BoxDecoration(color: color, shape: BoxShape.circle),
      );

  Widget _legendRow(Color color, String label) => Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          _dot(color),
          const SizedBox(width: 5),
          Text(label, style: const TextStyle(color: Colors.white60, fontSize: 12)),
        ],
      );

  @override
  Widget build(BuildContext context) {
    final withValue    = substances.where((s) => s.valor != null && s.valor! > 0).toList();
    final withoutValue = substances.where((s) => s.valor == null || s.valor! <= 0).toList();

    if (substances.isEmpty) return const SizedBox.shrink();

    // ── All unknown: full grey bar ────────────────────────────────────────────
    if (withValue.isEmpty) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Composición', style: _labelStyle),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Stack(
              alignment: Alignment.center,
              children: [
                Container(height: 10, color: _grey),
                const Text('?',
                    style: TextStyle(color: Colors.white54, fontSize: 8,
                        fontWeight: FontWeight.bold)),
              ],
            ),
          ),
          const SizedBox(height: 10),
          Wrap(
            spacing: 14, runSpacing: 6,
            children: withoutValue
                .map((s) => _legendRow(_grey, '${s.nombre}  —'))
                .toList(),
          ),
          const SizedBox(height: 20),
        ],
      );
    }

    // ── Has values: use pesoMg as total if available ──────────────────────────
    final knownTotal = withValue.fold(0.0, (sum, s) => sum + s.valor!);
    final total      = (pesoMg != null && pesoMg! > knownTotal) ? pesoMg! : knownTotal;
    final filler     = total - knownTotal;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Composición', style: _labelStyle),
        const SizedBox(height: 10),
        ClipRRect(
          borderRadius: BorderRadius.circular(4),
          child: Row(
            children: [
              ...withValue.asMap().entries.map((e) {
                final flex = ((e.value.valor! / total) * 1000).round().clamp(1, 1000);
                return Expanded(
                  flex: flex,
                  child: Container(height: 10, color: _palette[e.key % _palette.length]),
                );
              }),
              if (filler > 0)
                Expanded(
                  flex: ((filler / total) * 1000).round().clamp(1, 1000),
                  child: Container(height: 10, color: _grey),
                ),
            ],
          ),
        ),
        const SizedBox(height: 10),
        Wrap(
          spacing: 14, runSpacing: 6,
          children: [
            ...withValue.asMap().entries.map((e) {
              final pct = (e.value.valor! / total * 100).toStringAsFixed(1);
              return _legendRow(_palette[e.key % _palette.length],
                  '${e.value.nombre}  $pct%');
            }),
            if (filler > 0)
              _legendRow(_grey,
                  'Excipientes  ${(filler / total * 100).toStringAsFixed(1)}%'),
            ...withoutValue.map((s) => _legendRow(_grey, '${s.nombre}  —')),
          ],
        ),
        const SizedBox(height: 20),
      ],
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
