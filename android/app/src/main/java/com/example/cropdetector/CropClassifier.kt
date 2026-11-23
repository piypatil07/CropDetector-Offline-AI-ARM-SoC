package com.example.cropdetector

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import org.json.JSONObject
import org.json.JSONArray
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.security.MessageDigest
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp

/** All methods must be called from the same background worker. */
class CropClassifier(
    private val assets: AssetManager,
    val config: JSONObject,
    val precision: String,
    threads: Int = 4
) : Closeable {
    companion object {
        const val ROOT = "multiclass/"
        const val SIZE = 224
        const val CLASSES = 15
        const val PIXEL_VALUES = SIZE * SIZE * 3
        const val AUTO = "Automatic"
        const val AWARE = "Crop-Aware"

        fun readConfig(assets: AssetManager): JSONObject {
            val cfg = JSONObject(assets.open(ROOT + "model_config.json")
                .bufferedReader().use { it.readText() })
            require(cfg.getString("output_type") == "logits")
            require(cfg.getInt("num_classes") == CLASSES)
            require(cfg.getJSONArray("input_shape").let { a ->
                IntArray(a.length()) { a.getInt(it) }
            }.contentEquals(intArrayOf(1, SIZE, SIZE, 3)))
            require(cfg.getJSONArray("input_pixel_range").let {
                it.length() == 2 && it.getInt(0) == 0 && it.getInt(1) == 255
            })
            for (key in listOf("class_names", "display_names", "class_crops")) {
                require(cfg.getJSONArray(key).length() == CLASSES) { "Invalid $key" }
            }
            return cfg
        }

        fun strings(array: JSONArray): List<String> =
            List(array.length()) { array.getString(it) }

        fun argmax(values: FloatArray): Int =
            values.indices.maxByOrNull { values[it] } ?: error("Empty logits")

        fun bitmapToPixels(bitmap: Bitmap): FloatArray {
            val resized = Bitmap.createScaledBitmap(bitmap, SIZE, SIZE, true)
            val packed = IntArray(SIZE * SIZE)
            resized.getPixels(packed, 0, SIZE, 0, 0, SIZE, SIZE)
            val rgb = FloatArray(PIXEL_VALUES)
            for (i in packed.indices) {
                rgb[3 * i] = Color.red(packed[i]).toFloat()
                rgb[3 * i + 1] = Color.green(packed[i]).toFloat()
                rgb[3 * i + 2] = Color.blue(packed[i]).toFloat()
            }
            if (resized !== bitmap) resized.recycle()
            return rgb
        }

        /** Test the exact exported tensors; no image decoding or resizing here. */
        fun runParityTests(assets: AssetManager, config: JSONObject): Map<String, Boolean> {
            val cases = JSONArray(assets.open(ROOT + "golden_cases.json")
                .bufferedReader().use { it.readText() })
            require(cases.length() == 4) { "Expected four golden test vectors" }
            val results = linkedMapOf<String, Boolean>()
            for (format in listOf("FP32", "FP16", "INT8")) {
                try {
                    CropClassifier(assets, config, format, 4).use { classifier ->
                        // Provisional engineering smoke-test tolerances, not universal limits.
                        val tolerance = if (format == "INT8") {
                            2.0 * classifier.outputScale + 1e-5
                        } else 0.05
                        var passed = 0
                        for (i in 0 until cases.length()) {
                            val test = cases.getJSONObject(i)
                            val bytes = assets.open(ROOT + test.getString("input_file"))
                                .use { it.readBytes() }
                            require(bytes.size == PIXEL_VALUES * 4) { "Bad golden input size" }
                            val reader = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                            val rgb = FloatArray(PIXEL_VALUES)
                            reader.asFloatBuffer().get(rgb)
                            val expected = test.getJSONObject("expected_outputs").getJSONObject(format)
                            val expectedArray = expected.getJSONArray("logits")
                            require(expectedArray.length() == CLASSES)
                            val reference = FloatArray(CLASSES) { expectedArray.getDouble(it).toFloat() }
                            val actual = classifier.infer(rgb).logits
                            val errors = actual.indices.map { abs(actual[it] - reference[it]).toDouble() }
                            val maxError = errors.maxOrNull() ?: Double.POSITIVE_INFINITY
                            val mae = errors.average()
                            val expectedTop = expected.getInt("predicted_class")
                            val actualTop = argmax(actual)
                            val sameTop = actualTop == expectedTop
                            val pass = maxError <= tolerance && sameTop
                            if (pass) passed++
                            Log.i("CropParity", String.format(Locale.US,
                                "%s %s maxAbs=%.6f MAE=%.6f tolerance=%.6f expected=%d actual=%d top1Match=%s %s",
                                format, test.getString("crop"), maxError, mae, tolerance,
                                expectedTop, actualTop, sameTop, if (pass) "PASS" else "REVIEW"))
                            if (!pass) {
                                Log.i("CropParity", "Expected: ${reference.contentToString()}")
                                Log.i("CropParity", "Actual: ${actual.contentToString()}")
                            }
                        }
                        results[format] = passed == cases.length()
                        Log.i("CropParity", "$format SUMMARY $passed/${cases.length()} passed")
                    }
                } catch (e: Exception) {
                    results[format] = false
                    Log.e("CropParity", "$format MODEL/TEST ERROR", e)
                }
            }
            return results
        }
    }

    data class Inference(val logits: FloatArray, val latencyMs: Double,
                         val preparationMs: Double, val outputMs: Double)
    data class Candidate(val name: String, val score: Double)
    data class Prediction(
        val candidates: List<Candidate>, val accepted: Boolean,
        val threshold: Double, val latencyMs: Double
    )

    private val spec = config.getJSONObject("models").getJSONObject(precision)
    val cropOptions = listOf(AUTO) + strings(config.getJSONArray("supported_crops"))
    private val names = strings(config.getJSONArray("display_names"))
    private val crops = strings(config.getJSONArray("class_crops"))
    private val modelBuffer: ByteBuffer
    private val interpreter: Interpreter
    private val inputBuffer: ByteBuffer
    private val outputBuffer: ByteBuffer
    private val floatInputView: FloatBuffer?
    private val quantizedPixels: ByteArray?
    private val inputType: DataType
    private val outputType: DataType
    private val inputScale: Float
    private val inputZero: Int
    val outputScale: Float
    private val outputZero: Int

    init {
        val filename = spec.getString("filename")
        val bytes = assets.open(ROOT + filename).use { it.readBytes() }
        require(bytes.size.toLong() == spec.getLong("size_bytes")) { "Model file size mismatch: $filename" }
        val sha = MessageDigest.getInstance("SHA-256").digest(bytes)
            .joinToString("") { "%02x".format(it.toInt() and 0xff) }
        require(sha.equals(spec.getString("sha256"), ignoreCase = true)) { "Model SHA256 mismatch: $filename" }
        modelBuffer = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(bytes)
        modelBuffer.rewind()
        interpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(threads) })
        try {
            interpreter.allocateTensors()
            val input = interpreter.getInputTensor(0)
            val output = interpreter.getOutputTensor(0)
            require(interpreter.inputTensorCount == 1 && interpreter.outputTensorCount == 1)
            require(input.shape().contentEquals(intArrayOf(1, SIZE, SIZE, 3))) { "Unexpected input shape" }
            require(output.shape().contentEquals(intArrayOf(1, CLASSES))) { "Unexpected output shape" }
            inputType = input.dataType()
            outputType = output.dataType()
            val required = if (precision == "INT8") DataType.INT8 else DataType.FLOAT32
            require(inputType == required && outputType == required) { "Unexpected tensor data types" }
            inputScale = input.quantizationParams().scale
            inputZero = input.quantizationParams().zeroPoint
            outputScale = output.quantizationParams().scale
            outputZero = output.quantizationParams().zeroPoint
            if (required == DataType.INT8) {
                require(inputScale > 0f && outputScale > 0f)
                require(abs(inputScale.toDouble() - spec.getDouble("input_scale")) < 1e-6)
                require(inputZero == spec.getInt("input_zero_point"))
                require(abs(outputScale.toDouble() - spec.getDouble("output_scale")) < 1e-6)
                require(outputZero == spec.getInt("output_zero_point"))
            }
            inputBuffer = ByteBuffer.allocateDirect(input.numBytes()).order(ByteOrder.nativeOrder())
            outputBuffer = ByteBuffer.allocateDirect(output.numBytes()).order(ByteOrder.nativeOrder())
            floatInputView = if (inputType == DataType.FLOAT32) inputBuffer.asFloatBuffer() else null
            quantizedPixels = if (inputType == DataType.INT8) ByteArray(PIXEL_VALUES) else null
            Log.i("CropModel", "$precision loaded: input=${input.shape().contentToString()} $inputType scale=$inputScale zero=$inputZero; output=${output.shape().contentToString()} $outputType scale=$outputScale zero=$outputZero")
        } catch (e: Exception) {
            interpreter.close()
            throw e
        }
    }

    fun infer(rgb: FloatArray): Inference {
        require(rgb.size == PIXEL_VALUES)
        val prepStart = SystemClock.elapsedRealtimeNanos()
        inputBuffer.clear()
        if (inputType == DataType.INT8) {
            val dst = checkNotNull(quantizedPixels)
            if (inputScale == 1f && inputZero == -128) {
                for (i in rgb.indices) {
                    val pixel = rgb[i]
                    // Fast path for valid RGB inputs, preserving NumPy ties-to-even.
                    // Camera pixels are integers; resized golden tensors may be fractional.
                    val q = if (pixel >= 0f && pixel <= 255f) {
                        val whole = pixel.toInt()
                        val fraction = pixel - whole.toFloat()
                        val rounded = if (fraction > 0.5f ||
                            (fraction == 0.5f && (whole and 1) != 0)) whole + 1 else whole
                        rounded - 128
                    } else {
                        (Math.rint(pixel.toDouble()).toInt() - 128).coerceIn(-128, 127)
                    }
                    dst[i] = q.toByte()
                }
            } else {
                for (i in rgb.indices) {
                    dst[i] = (Math.rint(rgb[i].toDouble() / inputScale).toInt() + inputZero)
                        .coerceIn(-128, 127).toByte()
                }
            }
            inputBuffer.put(dst)
        } else {
            checkNotNull(floatInputView).apply { clear(); put(rgb) }
        }
        inputBuffer.rewind()
        outputBuffer.clear()
        val start = SystemClock.elapsedRealtimeNanos()
        val preparationMs = (start - prepStart) / 1e6
        interpreter.run(inputBuffer, outputBuffer)
        val outputStart = SystemClock.elapsedRealtimeNanos()
        val latencyMs = (outputStart - start) / 1e6
        outputBuffer.rewind()
        val logits = FloatArray(CLASSES) {
            if (outputType == DataType.INT8) (outputBuffer.get().toInt() - outputZero) * outputScale
            else outputBuffer.getFloat()
        }
        require(logits.all { it.isFinite() }) { "Non-finite model output" }
        return Inference(logits, latencyMs, preparationMs,
            (SystemClock.elapsedRealtimeNanos() - outputStart) / 1e6)
    }

    fun predict(bitmap: Bitmap, crop: String): Prediction {
        require(crop in cropOptions)
        val result = infer(bitmapToPixels(bitmap))
        val mode = if (crop == AUTO) AUTO else AWARE
        val temp = spec.getJSONObject("temperatures").getDouble(mode)
        val threshold = config.getJSONObject("confidence_thresholds").getDouble(mode)
        require(temp > 0 && temp.isFinite() && threshold in 0.0..1.0)
        val eligible = result.logits.indices.filter { crop == AUTO || crops[it] == crop }
        require(eligible.isNotEmpty()) { "No model classes for crop $crop" }
        val maximum = eligible.maxOf { result.logits[it].toDouble() / temp }
        val scores = eligible.map { exp(result.logits[it].toDouble() / temp - maximum) }
        val sum = scores.sum()
        val candidates = eligible.mapIndexed { i, cls -> Candidate(names[cls], scores[i] / sum) }
            .sortedByDescending { it.score }.take(3)
        return Prediction(candidates, candidates.first().score >= threshold, threshold, result.latencyMs)
    }

    override fun close() = interpreter.close()
}
