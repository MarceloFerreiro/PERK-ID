package com.pillid.ui

import android.graphics.Bitmap
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.pillid.R
import com.pillid.databinding.ActivityResultsBinding
import com.pillid.model.PillResult

class ResultsActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultsBinding

    companion object {
        // Temporary storage for passing data (avoid serialization issues)
        var queryBitmap: Bitmap? = null
        var results: List<PillResult> = emptyList()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupToolbar()
        displayResults()
    }

    private fun setupToolbar() {
        setSupportActionBar(binding.toolbar)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Search Results"
        binding.toolbar.setNavigationOnClickListener { onBackPressed() }
    }

    private fun displayResults() {
        // Display query image
        queryBitmap?.let {
            binding.queryImage.setImageBitmap(it)
        }

        // Setup results list
        binding.resultsRecyclerView.layoutManager = LinearLayoutManager(this)
        binding.resultsRecyclerView.adapter = ResultsAdapter(results)

        binding.resultsCount.text = "${results.size} matches found"
    }

    override fun onDestroy() {
        super.onDestroy()
        // Clear static references
        queryBitmap = null
        results = emptyList()
    }
}

class ResultsAdapter(private val results: List<PillResult>) :
    RecyclerView.Adapter<ResultsAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val rankText: TextView = view.findViewById(R.id.rankText)
        val scoreText: TextView = view.findViewById(R.id.scoreText)
        val scoreBar: View = view.findViewById(R.id.scoreBar)
        val colorText: TextView = view.findViewById(R.id.colorText)
        val logoText: TextView = view.findViewById(R.id.logoText)
        val procedenciaText: TextView = view.findViewById(R.id.procedenciaText)
        val fechaText: TextView = view.findViewById(R.id.fechaText)
        val pesoText: TextView = view.findViewById(R.id.pesoText)
        val diametroText: TextView = view.findViewById(R.id.diametroText)
        val sustanciasContainer: ViewGroup = view.findViewById(R.id.sustanciasContainer)
        val pillImage: ImageView = view.findViewById(R.id.pillImage)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_result, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val result = results[position]
        val meta = result.metadata

        // Rank and score
        holder.rankText.text = "#${result.rank}"
        val scorePercent = (result.score * 100).toInt()
        holder.scoreText.text = "$scorePercent%"

        // Score bar width
        val params = holder.scoreBar.layoutParams
        params.width = 0 // Will be set by weight
        holder.scoreBar.layoutParams = params
        holder.scoreBar.post {
            val parent = holder.scoreBar.parent as ViewGroup
            val maxWidth = parent.width - holder.scoreText.width - 16
            holder.scoreBar.layoutParams.width = (maxWidth * result.score).toInt()
            holder.scoreBar.requestLayout()
        }

        // Score bar color
        val colorRes = when {
            result.score >= 0.95f -> android.R.color.holo_green_dark
            result.score >= 0.85f -> android.R.color.holo_green_light
            result.score >= 0.70f -> android.R.color.holo_orange_light
            else -> android.R.color.holo_red_light
        }
        holder.scoreBar.setBackgroundResource(colorRes)

        // Metadata
        holder.colorText.text = meta.color ?: "-"
        holder.logoText.text = meta.logo ?: "-"
        holder.procedenciaText.text = meta.procedencia ?: "-"
        holder.fechaText.text = meta.fecha ?: "-"
        holder.pesoText.text = meta.pesoMg?.let { "${it.toInt()} mg" } ?: "-"
        holder.diametroText.text = meta.diametroMm?.let { "${it} mm" } ?: "-"

        // Sustancias
        holder.sustanciasContainer.removeAllViews()
        if (meta.sustancias.isNotEmpty()) {
            for (sus in meta.sustancias) {
                val susView = LayoutInflater.from(holder.itemView.context)
                    .inflate(R.layout.item_sustancia, holder.sustanciasContainer, false)
                val nombreText = susView.findViewById<TextView>(R.id.sustanciaNombre)
                val valorText = susView.findViewById<TextView>(R.id.sustanciaValor)

                nombreText.text = sus.nombre
                valorText.text = when {
                    sus.valor != null && sus.unidad != null -> "${sus.valor} ${sus.unidad}"
                    sus.valor != null -> "${sus.valor}"
                    else -> "-"
                }
                holder.sustanciasContainer.addView(susView)
            }
        } else {
            val noDataView = TextView(holder.itemView.context).apply {
                text = "No substance data"
                setTextColor(0xFF888888.toInt())
                textSize = 12f
            }
            holder.sustanciasContainer.addView(noDataView)
        }

        // Placeholder for pill image (actual images would need to be bundled or fetched)
        holder.pillImage.setImageResource(R.drawable.pill_placeholder)
    }

    override fun getItemCount() = results.size
}
