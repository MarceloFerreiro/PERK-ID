package com.pillid.core

import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import kotlin.math.max
import kotlin.math.min

object ImageUtils {

    const val CANONICAL_SIZE = 256

    /**
     * CLAHE on L channel of LAB to normalize illumination.
     * Port of correct_illumination() from Python.
     */
    fun correctIllumination(bgr: Mat): Mat {
        val lab = Mat()
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)

        val channels = ArrayList<Mat>()
        Core.split(lab, channels)

        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        clahe.apply(channels[0], channels[0])

        Core.merge(channels, lab)

        val result = Mat()
        Imgproc.cvtColor(lab, result, Imgproc.COLOR_Lab2BGR)

        // Clean up
        channels.forEach { it.release() }
        lab.release()

        return result
    }

    /**
     * Resize image maintaining aspect ratio and center on white canvas.
     */
    fun resizeToCanonical(image: Mat, targetSize: Int = CANONICAL_SIZE): Mat {
        val h = image.rows()
        val w = image.cols()
        val scale = targetSize.toFloat() / max(h, w)
        val newW = (w * scale).toInt()
        val newH = (h * scale).toInt()

        val resized = Mat()
        Imgproc.resize(image, resized, Size(newW.toDouble(), newH.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)

        // Create white canvas
        val canvas = Mat(targetSize, targetSize, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))

        // Center the resized image
        val offY = (targetSize - newH) / 2
        val offX = (targetSize - newW) / 2

        val roi = canvas.submat(offY, offY + newH, offX, offX + newW)
        resized.copyTo(roi)

        resized.release()
        return canvas
    }

    /**
     * Convert bitmap coordinates to Mat indices safely.
     */
    fun clamp(value: Int, minVal: Int, maxVal: Int): Int {
        return max(minVal, min(maxVal, value))
    }
}
