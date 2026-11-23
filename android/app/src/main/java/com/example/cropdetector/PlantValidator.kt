package com.example.cropdetector

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp

/**
 * Stage A, experimental plant-like / non-plant gate.
 * Run on the SAME background executor as CropClassifier. NOT thread safe.
 * Does not identify a supported crop or verify suitability for diagnosis.
 */
class PlantValidator(private val assets: AssetManager, threads: Int = 2) : Closeable {
    companion object {
        const val ROOT = "validator/"
        const val SIZE = 160
        const val PIXEL_VALUES = SIZE * SIZE * 3
        private const val LOG_TAG = "PlantGate"

        fun loadConfig(assets: AssetManager): JSONObject {
            val cfg = JSONObject(assets.open(ROOT + "stageA_deployment_config.json")
                .bufferedReader().use { it.readText() })
            require(cfg.getString("task") == "plant-like-versus-non-plant")
            val shape = cfg.getJSONArray("input_shape")
            require(shape.length() == 4 && shape.getInt(0) == 1 &&
                    shape.getInt(1) == SIZE && shape.getInt(2) == SIZE && shape.getInt(3) == 3)
            val threshold = cfg.getDouble("provisional_plant_acceptance_threshold")
            require(threshold in 0.0..1.0)
            return cfg
        }

        /**
         * Use the exact preprocessed float32 tensors from Colab. This test intentionally
         * bypasses camera decoding, rotation, resizing, and RGB extraction.
         * The golden vectors contain fractional pixel values; use ties-to-even rounding.
         */
        fun runParityTests(assets: AssetManager): Boolean {
            val cases = JSONArray(assets.open(ROOT + "golden_cases.json")
                .bufferedReader().use { it.readText() })
            require(cases.length() == 8) { "Expected eight Stage A golden inputs" }
            var passes = 0
            PlantValidator(assets, 2).use { model ->
                val tolerance = 2.0 * model.outputScale.toDouble() + 1e-5
                for (i in 0 until cases.length()) {
                    val item = cases.getJSONObject(i)
                    val bytes = assets.open(ROOT + item.getString("input_file"))
                        .use { it.readBytes() }
                    require(bytes.size == PIXEL_VALUES * 4) { "Bad Stage A golden tensor byte size" }
                    val floats = FloatArray(PIXEL_VALUES)
                    ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                        .asFloatBuffer().get(floats)
                    val expected = item.getJSONObject("expected_nonplant_logits")
                        .getDouble("INT8")
                    val output = model.infer(floats)
                    val expectedPlant = item.getInt("expected_truth") == 0
                    val close = abs(output.nonplantLogit - expected) <= tolerance
                    val match = output.accepted == expectedPlant
                    val pass = close && match
                    if (pass) passes++
                    Log.i("PlantParity", String.format(Locale.US,
                        "INT8 case=%d role=%s expected=%.6f actual=%.6f absError=%.6f " +
                        "tolerance=%.6f expectedPlant=%s actualPlant=%s %s",
                        i, item.getString("role"), expected, output.nonplantLogit,
                        abs(output.nonplantLogit - expected), tolerance,
                        expectedPlant, output.accepted, if (pass) "PASS" else "REVIEW"))
                }
            }
            Log.i("PlantParity", "INT8 SUMMARY $passes/${cases.length()} passed")
            return passes == cases.length()
        }

        fun pixelsFromBitmap(bitmap: Bitmap): FloatArray {
            val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
            try {
                val packed = IntArray(SIZE * SIZE)
                resized.getPixels(packed, 0, SIZE, 0, 0, SIZE, SIZE)
                val rgb = FloatArray(PIXEL_VALUES)
                for (i in packed.indices) {
                    rgb[3 * i] = Color.red(packed[i]).toFloat()
                    rgb[3 * i + 1] = Color.green(packed[i]).toFloat()
                    rgb[3 * i + 2] = Color.blue(packed[i]).toFloat()
                }
                return rgb
            } finally {
                if (resized !== bitmap) resized.recycle()
            }
        }

        private fun stablePlantScore(nonplantLogit: Double): Double {
            // sigmoid(-z), avoiding exp overflow for very large positive/negative z.
            return if (nonplantLogit >= 0.0) {
                val e = exp(-nonplantLogit)
                e / (1.0 + e)
            } else {
                1.0 / (1.0 + exp(nonplantLogit))
            }
        }
    }

    data class Result(
        val plantScore: Double,
        val accepted: Boolean,
        val nonplantLogit: Double,
        val threshold: Double,
        val preparationMs: Double,
        val interpreterMs: Double,
        val totalMs: Double
    )

    private val config = loadConfig(assets)
    val threshold: Double = config.getDouble("provisional_plant_acceptance_threshold")
    private val modelSpec = config.getJSONObject("models").getJSONObject("INT8")
    private val modelBuffer: ByteBuffer
    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val inputBytes = ByteArray(PIXEL_VALUES)
    private val inputScale: Float
    private val inputZero: Int
    val outputScale: Float
    private val outputZero: Int

    init {
        val filename = modelSpec.getString("filename")
        val bytes = assets.open(ROOT + filename).use { it.readBytes() }
        require(bytes.size.toLong() == modelSpec.getLong("size_bytes")) { "Stage A model byte count mismatch" }
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        require(sha.equals(modelSpec.getString("sha256"), ignoreCase = true)) {
            "Stage A model SHA-256 mismatch"
        }
        modelBuffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(bytes)
        modelBuffer.rewind()
        interpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(threads) })
        try {
            interpreter.allocateTensors()
            require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1)
            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            require(input.shape().contentEquals(intArrayOf(1, SIZE, SIZE, 3)))
            require(output.shape().contentEquals(intArrayOf(1, 1)))
            require(input.dataType() == DataType.INT8 && output.dataType() == DataType.INT8)
            inputScale = input.quantizationParams().scale
            inputZero = input.quantizationParams().zeroPoint
            outputScale = output.quantizationParams().scale
            outputZero = output.quantizationParams().zeroPoint
            require(inputScale > 0f && outputScale > 0f)
            val configIn = modelSpec.getJSONObject("input_quantization")
            val configOut = modelSpec.getJSONObject("output_quantization")
            require(abs(inputScale.toDouble() - configIn.getDouble("scale")) < 1e-6)
            require(inputZero == configIn.getInt("zero_point"))
            require(abs(outputScale.toDouble() - configOut.getDouble("scale")) < 1e-6)
            require(outputZero == configOut.getInt("zero_point"))
            inputBuffer = ByteBuffer.allocateDirect(input.numBytes()).order(ByteOrder.nativeOrder())
            outputBuffer = ByteBuffer.allocateDirect(output.numBytes()).order(ByteOrder.nativeOrder())
            Log.i("PlantModel", "Stage A INT8 loaded input=${input.shape().contentToString()} " +
                "scale=$inputScale zero=$inputZero output=${output.shape().contentToString()} " +
                "scale=$outputScale zero=$outputZero threshold=$threshold")
        } catch (e: Exception) {
            interpreter.close()
            throw e
        }
    }

    fun infer(rgb: FloatArray): Result {
        require(rgb.size == PIXEL_VALUES)
        val totalStart = SystemClock.elapsedRealtimeNanos()
        if (inputScale == 1f && inputZero == -128) {
            // Exact integer-RGB transform; ties-to-even for fractional golden pixels.
            for (i in rgb.indices) {
                val x = rgb[i]
                require(x.isFinite()) { "Non-finite Stage A input" }
                val rounded = if (x >= 0f && x <= 255f) {
                    val whole = x.toInt()
                    val frac = x - whole.toFloat()
                    if (frac > 0.5f || (frac == 0.5f && (whole and 1) != 0)) whole + 1 else whole
                } else {
                    Math.rint(x.toDouble()).toInt().coerceIn(0, 255)
                }
                inputBytes[i] = (rounded - 128).toByte()
            }
        } else {
            for (i in rgb.indices) {
                require(rgb[i].isFinite())
                inputBytes[i] = (Math.rint(rgb[i].toDouble() / inputScale).toInt() + inputZero)
                    .coerceIn(-128, 127).toByte()
            }
        }
        inputBuffer.clear()
        inputBuffer.put(inputBytes)
        inputBuffer.rewind()
        outputBuffer.clear()
        val interpreterStart = SystemClock.elapsedRealtimeNanos()
        interpreter.run(inputBuffer, outputBuffer)
        val afterRun = SystemClock.elapsedRealtimeNanos()
        outputBuffer.rewind()
        val raw = outputBuffer.get().toInt()
        val logit = ((raw - outputZero) * outputScale).toDouble()
        val score = stablePlantScore(logit)
        require(score.isFinite())
        return Result(score, score > threshold, logit, threshold,
            (interpreterStart - totalStart) / 1e6,
            (afterRun - interpreterStart) / 1e6,
            (SystemClock.elapsedRealtimeNanos() - totalStart) / 1e6)
    }

    /** Includes Bitmap resize + RGB pixel extraction; the Result timing excludes this preprocessing. */
    fun validate(bitmap: Bitmap): Result = infer(pixelsFromBitmap(bitmap))

    override fun close() = interpreter.close()
}
