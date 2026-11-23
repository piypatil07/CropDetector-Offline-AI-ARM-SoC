package com.example.cropdetector

import android.content.Context
import android.os.Build
import android.os.Debug
import android.os.PowerManager
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.CancellationException
import kotlin.math.ceil
import kotlin.math.sqrt
import kotlin.random.Random

/** Runs on the same single worker as the camera classifier. */
object CropBenchmark {
    private const val WARMUP = 8
    private const val RUNS = 40
    private const val ROUNDS = 3
    private data class Case(val pixels: FloatArray, val expected: Map<String, Int>)
    private data class Setting(val format: String, val threads: Int)
    private fun now() = SystemClock.elapsedRealtimeNanos()
    private fun ms(start: Long) = (now() - start) / 1e6
    private fun fmt(x: Double) = String.format(Locale.US, "%.4f", x)
    private fun median(x: List<Double>): Double {
        val s = x.sorted()
        val n = s.size
        return if (n % 2 == 0) (s[n/2-1] + s[n/2])/2 else s[n/2]
    }
    private fun thermal(context: Context): Int = if (Build.VERSION.SDK_INT >= 29) {
        (context.getSystemService(Context.POWER_SERVICE) as PowerManager).currentThermalStatus
    } else -1

    fun run(context: Context, config: JSONObject, formats: List<String>,
            cancelled: () -> Boolean, progress: (String) -> Unit): String {
        fun checkRunning() { if (cancelled()) throw CancellationException("Benchmark cancelled") }
        fun coolBeforeTrial() {
            val deadline = SystemClock.elapsedRealtime() + 120_000L
            while (thermal(context) >= 2) {
                checkRunning()
                if (SystemClock.elapsedRealtime() >= deadline) {
                    error("Device stayed at thermal status 2 or higher; cool phone and rerun")
                }
                progress("Cooling: thermal status ${thermal(context)}; waiting for 0 or 1")
                Thread.sleep(1000)
            }
        }
        fun checkThermal() {
            checkRunning()
            if (thermal(context) >= 3) error("Severe thermal throttling; cool phone and rerun")
        }
        val assets = context.assets
        val json = JSONArray(assets.open(CropClassifier.ROOT + "golden_cases.json")
            .bufferedReader().use { it.readText() })
        val cases = List(json.length()) { i ->
            val item = json.getJSONObject(i)
            val raw = assets.open(CropClassifier.ROOT + item.getString("input_file")).use { it.readBytes() }
            require(raw.size == CropClassifier.PIXEL_VALUES * 4)
            val rgb = FloatArray(CropClassifier.PIXEL_VALUES)
            ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(rgb)
            Case(rgb, formats.associateWith { format -> item.getJSONObject("expected_outputs")
                .getJSONObject(format).getInt("predicted_class") })
        }
        require(cases.size == 4)
        require(formats.isNotEmpty())
        val settings = formats.flatMap { f -> listOf(1, 2, 4).map { Setting(f, it) } }
        val folder = File(context.filesDir, "benchmarks_optimized").apply { mkdirs() }
        val summary = File(folder, "summary.csv")
        val rawFile = File(folder, "latencies.csv")
        summary.writeText("round,model,threads,init_ms,first_total_ms,first_run_ms,mean_total_ms,median_total_ms,p95_total_ms,std_total_ms,mean_run_ms,median_run_ms,p95_run_ms,throughput_per_s,top1_match_count,measured_runs,pss_before_kib,pss_loaded_kib,thermal_before,thermal_after,mean_prep_ms,mean_output_ms\n")
        rawFile.writeText("round,model,threads,run,input_index,total_ms,interpreter_ms,top1_match,prep_ms,output_ms\n")
        val means = linkedMapOf<Setting, MutableList<Double>>()
        var completed = 0
        val total = ROUNDS * settings.size
        for (round in 1..ROUNDS) {
            // Same seeded order on repeated experiments; vary order across rounds.
            for (setting in settings.shuffled(Random(2026 + round))) {
                checkRunning()
                coolBeforeTrial()
                val (format, threads) = setting
                progress("Benchmark ${completed + 1}/$total: $format, $threads thread(s)")
                val thermalBefore = thermal(context)
                val pssBefore = Debug.getPss()
                val initStart = now()
                CropClassifier(assets, config, format, threads).use { model ->
                    val initMs = ms(initStart)
                    val firstStart = now()
                    val first = model.infer(cases[0].pixels)
                    val firstTotal = ms(firstStart)
                    repeat(WARMUP) { i ->
                        checkThermal()
                        model.infer(cases[i % cases.size].pixels)
                    }
                    val pssLoaded = Debug.getPss()
                    val totals = mutableListOf<Double>()
                    val runs = mutableListOf<Double>()
                    val preps = mutableListOf<Double>()
                    val outputs = mutableListOf<Double>()
                    var matches = 0
                    val rawRows = StringBuilder()
                    repeat(RUNS) { i ->
                        checkRunning()
                        if (i % 4 == 0) checkThermal()
                        val index = i % cases.size
                        val input = cases[index]
                        val start = now()
                        val result = model.infer(input.pixels)
                        val elapsed = ms(start)
                        val match = CropClassifier.argmax(result.logits) == input.expected.getValue(format)
                        if (match) matches++
                        totals.add(elapsed)
                        runs.add(result.latencyMs)
                        preps.add(result.preparationMs)
                        outputs.add(result.outputMs)
                        rawRows.append("$round,$format,$threads,${i+1},$index,${fmt(elapsed)},${fmt(result.latencyMs)},$match,${fmt(result.preparationMs)},${fmt(result.outputMs)}\n")
                    }
                    checkThermal()
                    fun p95(values: List<Double>): Double = values.sorted()[ceil(0.95 * values.size).toInt()-1]
                    val mean = totals.average()
                    val std = sqrt(totals.map { (it-mean)*(it-mean) }.average())
                    val thermalAfter = thermal(context)
                    val row = listOf(round,format,threads,fmt(initMs),fmt(firstTotal),fmt(first.latencyMs),fmt(mean),fmt(median(totals)),fmt(p95(totals)),fmt(std),fmt(runs.average()),fmt(median(runs)),fmt(p95(runs)),fmt(1000/mean),matches,RUNS,pssBefore,pssLoaded,thermalBefore,thermalAfter,fmt(preps.average()),fmt(outputs.average())).joinToString(",")
                    rawFile.appendText(rawRows.toString())
                    summary.appendText(row + "\n")
                    means.getOrPut(setting) { mutableListOf() }.add(mean)
                    Log.i("CropBenchmark", "round=$round model=$format threads=$threads meanTotalMs=${fmt(mean)} p95TotalMs=${fmt(p95(totals))} meanRunMs=${fmt(runs.average())} prepMs=${fmt(preps.average())} outputMs=${fmt(outputs.average())} top1=$matches/$RUNS pssKiB=$pssLoaded thermal=$thermalBefore->$thermalAfter")
                }
                completed++
            }
        }
        val best = means.minByOrNull { median(it.value) }!!
        Log.i("CropBenchmark", "COMPLETE $completed/$total; fastest median round mean=${best.key.format}/${best.key.threads} threads ${fmt(median(best.value))} ms")
        return "Benchmark complete: $completed/$total configurations.\nFastest observed: ${best.key.format}, ${best.key.threads} thread(s), ${fmt(median(best.value))} ms (median of round means).\nCSV files saved in files/benchmarks_optimized."
    }
}
