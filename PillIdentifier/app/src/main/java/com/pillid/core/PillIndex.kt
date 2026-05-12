package com.pillid.core

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import com.pillid.model.PillMetadata
import com.pillid.model.PillResult
import com.pillid.model.SearchFilters
import com.pillid.model.Sustancia
import java.io.DataInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Pill image index for KNN similarity search.
 * Loads index from binary file and metadata from JSON.
 */
class PillIndex(context: Context) {

    private lateinit var ids: Array<String>
    private lateinit var vectors: Array<FloatArray>
    private lateinit var colors: Array<String>
    private lateinit var logos: Array<String>
    private lateinit var metadata: Map<String, PillMetadata>

    val size: Int get() = ids.size
    val dimensions: Int get() = if (vectors.isNotEmpty()) vectors[0].size else 0

    init {
        loadIndex(context)
        loadMetadata(context)
    }

    /**
     * Load binary index file.
     * Format:
     * - 4 bytes: N (number of pills)
     * - 4 bytes: D (dimension of vectors)
     * - N * D * 4 bytes: vectors (float32)
     * - N strings: ids (length-prefixed UTF-8)
     * - N strings: colors
     * - N strings: logos
     */
    private fun loadIndex(context: Context) {
        context.assets.open("index.bin").use { inputStream ->
            val dis = DataInputStream(inputStream)

            // Read header
            val n = dis.readInt()
            val d = dis.readInt()

            // Read vectors
            vectors = Array(n) { FloatArray(d) }
            val buffer = ByteArray(d * 4)
            for (i in 0 until n) {
                dis.readFully(buffer)
                val bb = ByteBuffer.wrap(buffer).order(ByteOrder.LITTLE_ENDIAN)
                for (j in 0 until d) {
                    vectors[i][j] = bb.getFloat()
                }
            }

            // Read ids
            ids = Array(n) { readString(dis) }

            // Read colors
            colors = Array(n) { readString(dis) }

            // Read logos
            logos = Array(n) { readString(dis) }
        }
    }

    private fun readString(dis: DataInputStream): String {
        val len = dis.readShort().toInt()
        if (len <= 0) return ""
        val bytes = ByteArray(len)
        dis.readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    /**
     * Load metadata JSON file.
     */
    private fun loadMetadata(context: Context) {
        context.assets.open("metadata.json").use { inputStream ->
            val json = inputStream.bufferedReader().readText()
            val gson = Gson()

            // Parse as Map<String, Any> first, then convert
            val type = object : TypeToken<Map<String, Map<String, Any>>>() {}.type
            val rawMap: Map<String, Map<String, Any>> = gson.fromJson(json, type)

            metadata = rawMap.mapValues { (_, v) -> parseMetadata(v) }
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun parseMetadata(raw: Map<String, Any>): PillMetadata {
        val sustanciasRaw = raw["sustancias"] as? List<Map<String, Any>> ?: emptyList()
        val sustancias = sustanciasRaw.map { s ->
            Sustancia(
                nombre = s["nombre"] as? String ?: "",
                valor = (s["valor"] as? Number)?.toFloat(),
                unidad = s["unidad"] as? String
            )
        }

        return PillMetadata(
            id = raw["id"] as? String ?: "",
            filename = raw["filename"] as? String ?: "",
            phash = raw["phash"] as? String,
            color = raw["color"] as? String,
            logo = raw["logo"] as? String,
            procedencia = raw["procedencia"] as? String,
            fecha = raw["fecha"] as? String,
            pesoMg = (raw["peso_mg"] as? Number)?.toFloat(),
            diametroMm = (raw["diametro_mm"] as? Number)?.toFloat(),
            divisible = raw["divisible"] as? String,
            sustancias = sustancias
        )
    }

    /**
     * Search for similar pills using cosine similarity.
     *
     * @param queryVec 136-dimensional feature vector
     * @param topK number of results to return
     * @param filters optional color/logo filters
     * @param weights optional feature weights
     * @return list of top-K similar pills
     */
    fun search(
        queryVec: FloatArray,
        topK: Int = 5,
        filters: SearchFilters = SearchFilters(),
        weights: Map<String, Float> = FeatureExtractor.DEFAULT_WEIGHTS
    ): List<PillResult> {

        // Apply weights
        val q = FeatureExtractor.applyWeights(queryVec, weights)

        // Build candidate indices (apply filters)
        val candidates = mutableListOf<Int>()
        for (i in ids.indices) {
            var pass = true
            if (filters.color != null && !colors[i].contains(filters.color, ignoreCase = true)) {
                pass = false
            }
            if (filters.logo != null && !logos[i].contains(filters.logo, ignoreCase = true)) {
                pass = false
            }
            if (pass) candidates.add(i)
        }

        if (candidates.isEmpty()) return emptyList()

        // Compute cosine similarity
        val qNorm = normalize(q)
        val similarities = candidates.map { idx ->
            val dbVec = FeatureExtractor.applyWeights(vectors[idx], weights)
            val dbNorm = normalize(dbVec)
            val sim = dotProduct(qNorm, dbNorm)
            Pair(idx, sim)
        }

        // Sort by similarity descending
        val sorted = similarities.sortedByDescending { it.second }

        // Return top-K results
        return sorted.take(topK).mapIndexed { rank, (idx, score) ->
            PillResult(
                rank = rank + 1,
                id = ids[idx],
                score = score,
                metadata = metadata[ids[idx]] ?: PillMetadata(
                    id = ids[idx],
                    filename = "",
                    phash = null,
                    color = colors[idx],
                    logo = logos[idx],
                    procedencia = null,
                    fecha = null,
                    pesoMg = null,
                    diametroMm = null,
                    divisible = null,
                    sustancias = emptyList()
                )
            )
        }
    }

    /**
     * Get unique colors for filter dropdown.
     */
    fun getUniqueColors(): List<String> {
        return colors.filter { it.isNotEmpty() }.distinct().sorted()
    }

    /**
     * Get unique logos for filter dropdown.
     */
    fun getUniqueLogos(): List<String> {
        return logos.filter { it.isNotEmpty() }.distinct().sorted()
    }

    private fun normalize(vec: FloatArray): FloatArray {
        val norm = sqrt(vec.map { it * it }.sum())
        return if (norm > 1e-10f) vec.map { it / norm }.toFloatArray() else vec
    }

    private fun dotProduct(a: FloatArray, b: FloatArray): Float {
        var sum = 0f
        for (i in a.indices) {
            sum += a[i] * b[i]
        }
        return sum
    }
}
