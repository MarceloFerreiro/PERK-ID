package com.pillid.core

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.*

/**
 * Port of feature extraction from Python extract_features.py.
 * Extracts 136-dimensional feature vector from canonical pill image.
 *
 * Feature breakdown:
 * - Color histogram (HSV): 34 dims (18 H + 8 S + 8 V)
 * - Shape (Hu moments + geometric): 12 dims
 * - Texture (LBP): 18 dims
 * - Edges (orientation histogram): 8 dims
 * - Perceptual hash: 64 dims
 * Total: 136 dims
 */
object FeatureExtractor {

    private const val HIST_BINS_H = 18
    private const val HIST_BINS_S = 8
    private const val HIST_BINS_V = 8
    private const val LBP_RADIUS = 2
    private const val LBP_POINTS = 16
    private const val PHASH_SIZE = 8
    private const val EDGE_BINS = 8
    private const val CANONICAL_SIZE = 256

    // Feature block offsets (for weighted search)
    val FEAT_OFFSETS = mapOf(
        "color_hist" to Pair(0, 34),
        "shape" to Pair(34, 46),
        "lbp" to Pair(46, 64),
        "edges" to Pair(64, 72),
        "phash" to Pair(72, 136)
    )

    // Default weights
    val DEFAULT_WEIGHTS = mapOf(
        "color_hist" to 2.0f,
        "shape" to 1.5f,
        "lbp" to 1.0f,
        "edges" to 0.8f,
        "phash" to 1.0f
    )

    /**
     * Extract full 136-dim feature vector from canonical BGR image.
     */
    fun extract(canonicalBgr: Mat): FloatArray {
        val colorHist = extractColorHistogram(canonicalBgr)   // 34
        val shape = extractShape(canonicalBgr)                 // 12
        val lbp = extractLbpTexture(canonicalBgr)              // 18
        val edges = extractEdges(canonicalBgr)                 // 8
        val phash = extractPhash(canonicalBgr)                 // 64

        // Concatenate all features
        return colorHist + shape + lbp + edges + phash
    }

    /**
     * HSV color histogram weighted by pill mask (excludes white background).
     */
    private fun extractColorHistogram(canonicalBgr: Mat): FloatArray {
        val hsv = Mat()
        Imgproc.cvtColor(canonicalBgr, hsv, Imgproc.COLOR_BGR2HSV)

        // Create pill mask (exclude white background: V>245, S<15)
        val channels = ArrayList<Mat>()
        Core.split(hsv, channels)

        val vMask = Mat()
        val sMask = Mat()
        Core.compare(channels[2], Scalar(245.0), vMask, Core.CMP_LE)  // V <= 245
        Core.compare(channels[1], Scalar(15.0), sMask, Core.CMP_GE)   // S >= 15
        val pillMask = Mat()
        Core.bitwise_or(vMask, sMask, pillMask)

        // Calculate histograms
        val hHist = Mat()
        val sHist = Mat()
        val vHist = Mat()

        Imgproc.calcHist(
            listOf(hsv), MatOfInt(0), pillMask, hHist,
            MatOfInt(HIST_BINS_H), MatOfFloat(0f, 180f)
        )
        Imgproc.calcHist(
            listOf(hsv), MatOfInt(1), pillMask, sHist,
            MatOfInt(HIST_BINS_S), MatOfFloat(0f, 256f)
        )
        Imgproc.calcHist(
            listOf(hsv), MatOfInt(2), pillMask, vHist,
            MatOfInt(HIST_BINS_V), MatOfFloat(0f, 256f)
        )

        // Convert to arrays and normalize
        val hArr = FloatArray(HIST_BINS_H)
        val sArr = FloatArray(HIST_BINS_S)
        val vArr = FloatArray(HIST_BINS_V)

        for (i in 0 until HIST_BINS_H) hArr[i] = hHist.get(i, 0)[0].toFloat()
        for (i in 0 until HIST_BINS_S) sArr[i] = sHist.get(i, 0)[0].toFloat()
        for (i in 0 until HIST_BINS_V) vArr[i] = vHist.get(i, 0)[0].toFloat()

        val combined = hArr + sArr + vArr
        val total = combined.sum()
        val result = if (total > 0) combined.map { it / total }.toFloatArray() else combined

        // Cleanup
        hsv.release()
        channels.forEach { it.release() }
        vMask.release()
        sMask.release()
        pillMask.release()
        hHist.release()
        sHist.release()
        vHist.release()

        return result
    }

    /**
     * Shape features: 7 Hu moments + 5 geometric descriptors.
     */
    private fun extractShape(canonicalBgr: Mat): FloatArray {
        val gray = Mat()
        Imgproc.cvtColor(canonicalBgr, gray, Imgproc.COLOR_BGR2GRAY)

        val binary = Mat()
        Imgproc.threshold(gray, binary, 240.0, 255.0, Imgproc.THRESH_BINARY_INV)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(binary, binary, Imgproc.MORPH_CLOSE, kernel)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        if (contours.isEmpty()) {
            gray.release()
            binary.release()
            kernel.release()
            hierarchy.release()
            return FloatArray(12) { 0f }
        }

        // Find largest contour
        val cnt = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return FloatArray(12) { 0f }
        val area = Imgproc.contourArea(cnt)

        if (area == 0.0) {
            gray.release()
            binary.release()
            kernel.release()
            hierarchy.release()
            return FloatArray(12) { 0f }
        }

        // Bounding rect and hull
        val boundRect = Imgproc.boundingRect(cnt)
        val hull = MatOfInt()
        Imgproc.convexHull(cnt, hull)
        val hullPoints = MatOfPoint()
        val hullIndices = hull.toArray()
        val cntArray = cnt.toArray()
        hullPoints.fromArray(*hullIndices.map { cntArray[it] }.toTypedArray())
        val hullArea = Imgproc.contourArea(hullPoints)

        // Geometric descriptors
        val aspectRatio = if (boundRect.height > 0) boundRect.width.toFloat() / boundRect.height else 1f
        val rectArea = boundRect.width * boundRect.height
        val extent = if (rectArea > 0) area.toFloat() / rectArea else 0f
        val solidity = if (hullArea > 0) area.toFloat() / hullArea.toFloat() else 0f
        val equivDiameter = sqrt(4 * area / PI).toFloat() / CANONICAL_SIZE
        val widthRatio = boundRect.width.toFloat() / CANONICAL_SIZE

        // Hu moments
        val moments = Imgproc.moments(cnt)
        val huMoments = Mat()
        Imgproc.HuMoments(moments, huMoments)

        val huNorm = FloatArray(7)
        for (i in 0 until 7) {
            val hu = huMoments.get(i, 0)[0]
            // Log transform and normalize
            huNorm[i] = (-sign(hu) * log10(abs(hu) + 1e-10) / 10.0).toFloat()
        }

        val geoVec = floatArrayOf(aspectRatio, extent, solidity, equivDiameter, widthRatio)

        // Cleanup
        gray.release()
        binary.release()
        kernel.release()
        hierarchy.release()
        hull.release()
        hullPoints.release()
        huMoments.release()

        return huNorm + geoVec  // 12 dims
    }

    /**
     * LBP (Local Binary Pattern) texture histogram.
     */
    private fun extractLbpTexture(canonicalBgr: Mat): FloatArray {
        val gray = Mat()
        Imgproc.cvtColor(canonicalBgr, gray, Imgproc.COLOR_BGR2GRAY)

        val h = gray.rows()
        val w = gray.cols()

        // Calculate LBP
        val lbp = Mat.zeros(h, w, CvType.CV_8UC1)
        val angles = (0 until LBP_POINTS).map { 2 * PI * it / LBP_POINTS }
        val neighbors = angles.map { a ->
            Pair(
                (LBP_RADIUS * sin(a)).roundToInt(),
                (LBP_RADIUS * cos(a)).roundToInt()
            )
        }

        for ((dy, dx) in neighbors) {
            val shifted = Mat()
            // Roll operation
            val M = Mat.eye(2, 3, CvType.CV_64F)
            M.put(0, 2, dx.toDouble())
            M.put(1, 2, dy.toDouble())
            Imgproc.warpAffine(gray, shifted, M, gray.size(), Imgproc.INTER_NEAREST, Core.BORDER_WRAP)

            // Compare and accumulate
            val comparison = Mat()
            Core.compare(gray, shifted, comparison, Core.CMP_GE)
            comparison.convertTo(comparison, CvType.CV_8UC1, 1.0 / 255.0)

            // Shift left and OR
            val lbpNew = Mat()
            Core.multiply(lbp, Scalar(2.0), lbpNew)
            Core.add(lbpNew, comparison, lbp)

            shifted.release()
            comparison.release()
            lbpNew.release()
            M.release()
        }

        // Create mask for pill area (gray < 240)
        val mask = Mat()
        Core.compare(gray, Scalar(240.0), mask, Core.CMP_LT)

        // Calculate histogram
        val nBins = LBP_POINTS + 2
        val hist = FloatArray(nBins) { 0f }

        // Manual histogram calculation with mask
        for (y in 0 until h) {
            for (x in 0 until w) {
                if (mask.get(y, x)[0] > 0) {
                    val bin = (lbp.get(y, x)[0].toInt() * nBins / 256).coerceIn(0, nBins - 1)
                    hist[bin]++
                }
            }
        }

        // Normalize
        val total = hist.sum()
        val result = if (total > 0) hist.map { it / total }.toFloatArray() else hist

        gray.release()
        lbp.release()
        mask.release()

        return result  // 18 dims
    }

    /**
     * Edge orientation histogram (simplified HOG).
     */
    private fun extractEdges(canonicalBgr: Mat): FloatArray {
        val gray = Mat()
        Imgproc.cvtColor(canonicalBgr, gray, Imgproc.COLOR_BGR2GRAY)

        val gx = Mat()
        val gy = Mat()
        Imgproc.Sobel(gray, gx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(gray, gy, CvType.CV_32F, 0, 1, 3)

        val mag = Mat()
        val angle = Mat()
        Core.cartToPolar(gx, gy, mag, angle, true)  // angleInDegrees=true

        // Histogram of orientations weighted by magnitude
        val hist = FloatArray(EDGE_BINS) { 0f }
        val h = gray.rows()
        val w = gray.cols()

        for (y in 0 until h) {
            for (x in 0 until w) {
                val m = mag.get(y, x)[0].toFloat()
                if (m > 10) {
                    val a = angle.get(y, x)[0]
                    val bin = ((a / 360.0) * EDGE_BINS).toInt().coerceIn(0, EDGE_BINS - 1)
                    hist[bin] += m
                }
            }
        }

        // Normalize
        val total = hist.sum()
        val result = if (total > 0) hist.map { it / total }.toFloatArray() else hist

        gray.release()
        gx.release()
        gy.release()
        mag.release()
        angle.release()

        return result  // 8 dims
    }

    /**
     * Perceptual hash (pHash) using DCT.
     * Returns 64-dim binary vector (0/1 values).
     */
    private fun extractPhash(canonicalBgr: Mat): FloatArray {
        val gray = Mat()
        Imgproc.cvtColor(canonicalBgr, gray, Imgproc.COLOR_BGR2GRAY)

        // Resize to 32x32 for DCT
        val small = Mat()
        Imgproc.resize(gray, small, Size(32.0, 32.0), 0.0, 0.0, Imgproc.INTER_AREA)

        // Convert to float for DCT
        val smallFloat = Mat()
        small.convertTo(smallFloat, CvType.CV_32F)

        // Apply DCT
        val dct = Mat()
        Core.dct(smallFloat, dct)

        // Take top-left 8x8 (low frequencies)
        val dctLow = dct.submat(0, PHASH_SIZE, 0, PHASH_SIZE)

        // Calculate mean (excluding DC component)
        var sum = 0.0
        var count = 0
        for (y in 0 until PHASH_SIZE) {
            for (x in 0 until PHASH_SIZE) {
                if (y > 0 || x > 0) {  // Skip DC
                    sum += dctLow.get(y, x)[0]
                    count++
                }
            }
        }
        val mean = sum / count

        // Create 64-bit hash as float array
        val hash = FloatArray(64)
        var idx = 0
        for (y in 0 until PHASH_SIZE) {
            for (x in 0 until PHASH_SIZE) {
                hash[idx++] = if (dctLow.get(y, x)[0] > mean) 1f else 0f
            }
        }

        gray.release()
        small.release()
        smallFloat.release()
        dct.release()

        return hash  // 64 dims
    }

    /**
     * Apply weights to feature vector for weighted search.
     */
    fun applyWeights(features: FloatArray, weights: Map<String, Float> = DEFAULT_WEIGHTS): FloatArray {
        val weighted = features.copyOf()
        for ((block, range) in FEAT_OFFSETS) {
            val weight = weights[block] ?: 1f
            for (i in range.first until range.second) {
                weighted[i] *= weight
            }
        }
        return weighted
    }
}
