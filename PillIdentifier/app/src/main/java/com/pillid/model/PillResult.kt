package com.pillid.model

data class PillResult(
    val rank: Int,
    val id: String,
    val score: Float,
    val metadata: PillMetadata
)

data class PillMetadata(
    val id: String,
    val filename: String,
    val phash: String?,
    val color: String?,
    val logo: String?,
    val procedencia: String?,
    val fecha: String?,
    val pesoMg: Float?,
    val diametroMm: Float?,
    val divisible: String?,
    val sustancias: List<Sustancia>
)

data class Sustancia(
    val nombre: String,
    val valor: Float?,
    val unidad: String?
)

data class SearchFilters(
    val color: String? = null,
    val logo: String? = null
)
