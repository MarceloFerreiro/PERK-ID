package com.pillid.core

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Port of segment_pill() from Python extract_features.py.
 * Segments the pill from background using GrabCut + morphology.
 */
object PillSegmenter {

    private const val MIN_PILL_AREA = 0.05f

    /**
     * Segment pill and return canonical image (256x256 on white background).
     */
    fun segment(bgr: Mat): Mat {
        val h = bgr.rows()
        val w = bgr.cols()

        val gray = Mat()
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        val kernelLarge = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(11.0, 11.0))

        // Step 1: Canny to find initial rect
        val (rect, cannyContour) = findCannyRect(gray, w, h, kernel)

        // If Canny didn't find anything useful, use central rect
        val useRect = rect ?: Rect(
            (w * 0.15).toInt(),
            (h * 0.15).toInt(),
            (w * 0.70).toInt(),
            (h * 0.70).toInt()
        )

        // Step 2: GrabCut
        var maskFg = grabCutSegment(bgr, useRect, h, w)

        // Close internal holes
        Imgproc.morphologyEx(maskFg, maskFg, Imgproc.MORPH_CLOSE, kernelLarge, Point(-1.0, -1.0), 2)

        // Step 3: Flood fill from corners
        maskFg = floodFillBackground(maskFg)

        // Smooth edges
        Imgproc.morphologyEx(maskFg, maskFg, Imgproc.MORPH_OPEN, kernel, Point(-1.0, -1.0), 1)
        maskFg = keepBestCentralComponent(maskFg, w, h)

        var pillAreaRatio = Core.countNonZero(maskFg).toFloat() / (h * w)

        // Step 4: Fallback - fill Canny contour directly
        if (pillAreaRatio < MIN_PILL_AREA && cannyContour != null) {
            maskFg.setTo(Scalar(0.0))
            Imgproc.drawContours(maskFg, listOf(cannyContour), -1, Scalar(255.0), -1)
            Imgproc.morphologyEx(maskFg, maskFg, Imgproc.MORPH_CLOSE, kernelLarge, Point(-1.0, -1.0), 2)
            pillAreaRatio = Core.countNonZero(maskFg).toFloat() / (h * w)
        }

        // Step 5: Last resort - Otsu
        if (pillAreaRatio < MIN_PILL_AREA) {
            Imgproc.threshold(gray, maskFg, 0.0, 255.0, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU)
            maskFg = keepBestCentralComponent(maskFg, w, h)
        }

        // Crop bounding box
        val canonical = cropAndNormalize(bgr, maskFg)

        // Cleanup
        gray.release()
        kernel.release()
        kernelLarge.release()
        maskFg.release()

        return canonical
    }

    private fun findCannyRect(gray: Mat, w: Int, h: Int, kernel: Mat): Pair<Rect?, MatOfPoint?> {
        val blurred = Mat()
        Imgproc.GaussianBlur(gray, blurred, Size(7.0, 7.0), 0.0)

        val median = Core.mean(blurred).`val`[0]
        val sigmaLow = 0.33
        val lo = max(0.0, median * (1.0 - sigmaLow)).toInt()
        val hi = min(255.0, median * (1.0 + sigmaLow)).toInt()

        val edges = Mat()
        Imgproc.Canny(blurred, edges, lo.toDouble(), hi.toDouble())
        Imgproc.dilate(edges, edges, kernel, Point(-1.0, -1.0), 2)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)

        blurred.release()
        edges.release()
        hierarchy.release()

        if (contours.isEmpty()) return Pair(null, null)

        val cxImg = w / 2.0
        val cyImg = h / 2.0

        var bestContour: MatOfPoint? = null
        var bestScore = -1.0

        for (cnt in contours) {
            val area = Imgproc.contourArea(cnt)
            if (area < 500) continue

            val moments = Imgproc.moments(cnt)
            if (moments.m00 == 0.0) continue

            val cx = moments.m10 / moments.m00
            val cy = moments.m01 / moments.m00
            val dist = hypot(cx - cxImg, cy - cyImg)
            val score = area / (1.0 + dist)

            if (score > bestScore) {
                bestScore = score
                bestContour = cnt
            }
        }

        if (bestContour == null) return Pair(null, null)

        val boundRect = Imgproc.boundingRect(bestContour)
        val padX = max(4, (boundRect.width * 0.08).toInt())
        val padY = max(4, (boundRect.height * 0.08).toInt())

        val x1 = max(0, boundRect.x - padX)
        val y1 = max(0, boundRect.y - padY)
        val x2 = min(w, boundRect.x + boundRect.width + padX)
        val y2 = min(h, boundRect.y + boundRect.height + padY)

        return Pair(Rect(x1, y1, x2 - x1, y2 - y1), bestContour)
    }

    private fun grabCutSegment(bgr: Mat, rect: Rect, h: Int, w: Int): Mat {
        val maskGc = Mat.zeros(h, w, CvType.CV_8UC1)
        val bgModel = Mat()
        val fgModel = Mat()

        try {
            Imgproc.grabCut(bgr, maskGc, rect, bgModel, fgModel, 5, Imgproc.GC_INIT_WITH_RECT)

            // Convert to binary mask (FGD or PR_FGD = 255)
            val maskFg = Mat()
            Core.compare(maskGc, Scalar(Imgproc.GC_FGD.toDouble()), maskFg, Core.CMP_EQ)
            val maskPrFg = Mat()
            Core.compare(maskGc, Scalar(Imgproc.GC_PR_FGD.toDouble()), maskPrFg, Core.CMP_EQ)
            Core.bitwise_or(maskFg, maskPrFg, maskFg)

            maskPrFg.release()
            bgModel.release()
            fgModel.release()
            maskGc.release()

            return maskFg
        } catch (e: Exception) {
            bgModel.release()
            fgModel.release()
            maskGc.release()
            return Mat.zeros(h, w, CvType.CV_8UC1)
        }
    }

    private fun floodFillBackground(mask: Mat): Mat {
        val inv = Mat()
        Core.bitwise_not(mask, inv)

        val flood = Mat()
        Core.copyMakeBorder(inv, flood, 1, 1, 1, 1, Core.BORDER_CONSTANT, Scalar(0.0))

        val maskFf = Mat.zeros(flood.rows() + 2, flood.cols() + 2, CvType.CV_8UC1)
        Imgproc.floodFill(flood, maskFf, Point(0.0, 0.0), Scalar(128.0))

        // Remove border
        val floodCropped = flood.submat(1, flood.rows() - 1, 1, flood.cols() - 1)

        // Pixels reached by flood = background
        val bgReached = Mat()
        Core.compare(floodCropped, Scalar(128.0), bgReached, Core.CMP_EQ)

        val bgReachedInv = Mat()
        Core.bitwise_not(bgReached, bgReachedInv)

        val result = Mat()
        Core.bitwise_and(mask, bgReachedInv, result)

        inv.release()
        flood.release()
        maskFf.release()
        bgReached.release()
        bgReachedInv.release()

        return result
    }

    private fun keepBestCentralComponent(mask: Mat, w: Int, h: Int): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        val nComponents = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids)

        if (nComponents <= 1) {
            labels.release()
            stats.release()
            centroids.release()
            return mask
        }

        val cxImg = w / 2.0
        val cyImg = h / 2.0
        var bestIdx = 1
        var bestScore = -1.0

        for (i in 1 until nComponents) {
            val area = stats.get(i, Imgproc.CC_STAT_AREA)[0]
            if (area < 200) continue

            val cx = centroids.get(i, 0)[0]
            val cy = centroids.get(i, 1)[0]
            val dist = hypot(cx - cxImg, cy - cyImg)
            val score = area / (1.0 + dist)

            if (score > bestScore) {
                bestScore = score
                bestIdx = i
            }
        }

        val result = Mat()
        Core.compare(labels, Scalar(bestIdx.toDouble()), result, Core.CMP_EQ)
        result.convertTo(result, CvType.CV_8UC1, 255.0)

        labels.release()
        stats.release()
        centroids.release()

        return result
    }

    private fun cropAndNormalize(bgr: Mat, mask: Mat): Mat {
        val h = bgr.rows()
        val w = bgr.cols()

        val points = Mat()
        Core.findNonZero(mask, points)

        if (points.empty()) {
            points.release()
            return ImageUtils.resizeToCanonical(bgr)
        }

        val boundRect = Imgproc.boundingRect(points)
        points.release()

        val pad = (max(boundRect.width, boundRect.height) * 0.05).toInt()
        val x1 = max(0, boundRect.x - pad)
        val y1 = max(0, boundRect.y - pad)
        val x2 = min(w, boundRect.x + boundRect.width + pad)
        val y2 = min(h, boundRect.y + boundRect.height + pad)

        val crop = bgr.submat(y1, y2, x1, x2).clone()
        val maskCrop = mask.submat(y1, y2, x1, x2)

        // Set background to white
        val maskInv = Mat()
        Core.bitwise_not(maskCrop, maskInv)
        crop.setTo(Scalar(255.0, 255.0, 255.0), maskInv)
        maskInv.release()

        val canonical = ImageUtils.resizeToCanonical(crop)
        crop.release()

        return canonical
    }
}
