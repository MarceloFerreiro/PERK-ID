package com.pillid.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomsheet.BottomSheetBehavior
import com.pillid.core.FeatureExtractor
import com.pillid.core.ImageUtils
import com.pillid.core.PillIndex
import com.pillid.core.PillSegmenter
import com.pillid.databinding.ActivityMainBinding
import com.pillid.model.SearchFilters
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import java.io.InputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private var pillIndex: PillIndex? = null
    private var filterBottomSheet: BottomSheetBehavior<View>? = null

    private var selectedColor: String? = null
    private var selectedLogo: String? = null

    companion object {
        private const val TAG = "PillIdentifier"
        const val EXTRA_RESULTS = "results"
        const val EXTRA_QUERY_IMAGE = "query_image"
    }

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
        }
    }

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { processImageFromUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize OpenCV
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV init failed", Toast.LENGTH_LONG).show()
            return
        }
        Log.d(TAG, "OpenCV initialized successfully")

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Load index in background
        lifecycleScope.launch {
            loadIndex()
        }

        setupUI()
        checkCameraPermission()
    }

    private fun setupUI() {
        // Capture button
        binding.captureButton.setOnClickListener {
            takePhoto()
        }

        // Gallery button
        binding.galleryButton.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }

        // Filter button
        binding.filterButton.setOnClickListener {
            showFilterDialog()
        }

        // Setup filter bottom sheet
        filterBottomSheet = BottomSheetBehavior.from(binding.filterBottomSheet)
        filterBottomSheet?.state = BottomSheetBehavior.STATE_HIDDEN

        binding.applyFiltersButton.setOnClickListener {
            selectedColor = binding.colorInput.text.toString().takeIf { it.isNotBlank() }
            selectedLogo = binding.logoInput.text.toString().takeIf { it.isNotBlank() }
            filterBottomSheet?.state = BottomSheetBehavior.STATE_HIDDEN
            updateFilterBadge()
        }

        binding.clearFiltersButton.setOnClickListener {
            selectedColor = null
            selectedLogo = null
            binding.colorInput.text?.clear()
            binding.logoInput.text?.clear()
            filterBottomSheet?.state = BottomSheetBehavior.STATE_HIDDEN
            updateFilterBadge()
        }
    }

    private fun updateFilterBadge() {
        val hasFilters = selectedColor != null || selectedLogo != null
        binding.filterBadge.visibility = if (hasFilters) View.VISIBLE else View.GONE
    }

    private fun showFilterDialog() {
        filterBottomSheet?.state = BottomSheetBehavior.STATE_EXPANDED
    }

    private suspend fun loadIndex() {
        withContext(Dispatchers.IO) {
            try {
                pillIndex = PillIndex(this@MainActivity)
                Log.d(TAG, "Index loaded: ${pillIndex?.size} pills")
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load index", e)
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "Failed to load pill database", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.viewFinder.surfaceProvider)
                }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val imageCapture = imageCapture ?: return

        showLoading(true)

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                    image.close()
                    processImage(bitmap)
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed", exception)
                    showLoading(false)
                    Toast.makeText(this@MainActivity, "Capture failed", Toast.LENGTH_SHORT).show()
                }
            }
        )
    }

    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)

        // Rotate if needed
        val rotation = image.imageInfo.rotationDegrees
        return if (rotation != 0) {
            val matrix = Matrix()
            matrix.postRotate(rotation.toFloat())
            Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } else {
            bitmap
        }
    }

    private fun processImageFromUri(uri: Uri) {
        showLoading(true)

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                val inputStream: InputStream? = contentResolver.openInputStream(uri)
                val bitmap = BitmapFactory.decodeStream(inputStream)
                inputStream?.close()

                if (bitmap != null) {
                    withContext(Dispatchers.Main) {
                        processImage(bitmap)
                    }
                } else {
                    withContext(Dispatchers.Main) {
                        showLoading(false)
                        Toast.makeText(this@MainActivity, "Failed to load image", Toast.LENGTH_SHORT).show()
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Error loading image", e)
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    Toast.makeText(this@MainActivity, "Error loading image", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun processImage(bitmap: Bitmap) {
        lifecycleScope.launch(Dispatchers.Default) {
            try {
                val index = pillIndex
                if (index == null) {
                    withContext(Dispatchers.Main) {
                        showLoading(false)
                        Toast.makeText(this@MainActivity, "Index not loaded yet", Toast.LENGTH_SHORT).show()
                    }
                    return@launch
                }

                // Convert bitmap to Mat
                val mat = Mat()
                val bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                Utils.bitmapToMat(bmp32, mat)

                // BGR conversion (OpenCV uses BGR)
                org.opencv.imgproc.Imgproc.cvtColor(mat, mat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)

                // Process: CLAHE -> Segment -> Extract features
                val corrected = ImageUtils.correctIllumination(mat)
                val canonical = PillSegmenter.segment(corrected)
                val features = FeatureExtractor.extract(canonical)

                // Search
                val filters = SearchFilters(color = selectedColor, logo = selectedLogo)
                val results = index.search(features, topK = 5, filters = filters)

                // Convert canonical to bitmap for display
                val canonicalBitmap = Bitmap.createBitmap(
                    canonical.cols(), canonical.rows(), Bitmap.Config.ARGB_8888
                )
                org.opencv.imgproc.Imgproc.cvtColor(canonical, canonical, org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA)
                Utils.matToBitmap(canonical, canonicalBitmap)

                // Cleanup
                mat.release()
                corrected.release()
                canonical.release()

                withContext(Dispatchers.Main) {
                    showLoading(false)

                    if (results.isEmpty()) {
                        Toast.makeText(this@MainActivity, "No matches found", Toast.LENGTH_SHORT).show()
                    } else {
                        // Navigate to results
                        val intent = Intent(this@MainActivity, ResultsActivity::class.java)
                        ResultsActivity.queryBitmap = canonicalBitmap
                        ResultsActivity.results = results
                        startActivity(intent)
                    }
                }

            } catch (e: Exception) {
                Log.e(TAG, "Processing error", e)
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    Toast.makeText(this@MainActivity, "Processing failed: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }
        }
    }

    private fun showLoading(show: Boolean) {
        binding.loadingOverlay.visibility = if (show) View.VISIBLE else View.GONE
        binding.captureButton.isEnabled = !show
        binding.galleryButton.isEnabled = !show
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
