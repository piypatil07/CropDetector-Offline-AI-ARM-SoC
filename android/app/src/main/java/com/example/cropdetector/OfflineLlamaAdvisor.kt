package com.example.cropdetector

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import java.io.File
import java.io.Closeable
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Phase 10 — text-only, on-demand offline agricultural advisory.
 *
 * - ExecuTorch model stays OUTSIDE the APK, in app-specific files/models/.
 * - Loads/generates/closes the native LLM on a background thread.
 * - Never invokes the disease classifier; caller must pass confirmed crop and
 *   the existing classifier's accepted flag.
 * - This is an integration smoke-test component, NOT validated agronomic advice.
 */
class OfflineLlamaAdvisor(context: Context) : Closeable {
    companion object {
        private const val TAG = "CropLlama"
        private const val MODEL_NAME = "Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
        private const val TOKENIZER_NAME = "tokenizer.model"
        private val SUPPORTED_CROPS = setOf("Tomato", "Potato", "Maize", "Grape")
    }

    private val appContext = context.applicationContext
    private val worker = Executors.newSingleThreadExecutor()
    private val ui = Handler(Looper.getMainLooper())
    private val busy = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)

    private fun modelDirectory(): File {
        // App-owned directory. /data/local/tmp/llama is a shell staging directory,
        // not a dependable model path for an ordinary production Android app.
        val base = appContext.getExternalFilesDir(null) ?: appContext.filesDir
        return File(base, "models")
    }

    fun modelFilesPresent(): Boolean {
        val dir = modelDirectory()
        val model = File(dir, MODEL_NAME)
        val tokenizer = File(dir, TOKENIZER_NAME)
        return model.isFile && model.length() > 1_000_000_000L &&
               tokenizer.isFile && tokenizer.length() > 1_000_000L
    }

    /**
     * The caller MUST obtain crop from manual selection/confirmation, not the
     * experimental Stage B auto-recognizer. `diagnosisAccepted` is Stage C's
     * acceptance result; low-confidence labels must NOT be presented as facts.
     */
    fun generate(
        confirmedCrop: String,
        predictedCondition: String?,
        diagnosisAccepted: Boolean,
        onToken: (String) -> Unit,
        onComplete: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        if (closed.get()) {
            onError("Advisor is closed")
            return
        }
        if (confirmedCrop !in SUPPORTED_CROPS) {
            onError("Please confirm Tomato, Potato, Maize or Grape first")
            return
        }
        if (!busy.compareAndSet(false, true)) {
            onError("A Llama request is already running")
            return
        }

        worker.execute {
            try {
                val dir = modelDirectory()
                val modelFile = File(dir, MODEL_NAME)
                val tokenizerFile = File(dir, TOKENIZER_NAME)
                require(modelFilesPresent()) {
                    "Model files missing in ${dir.absolutePath}. Copy the verified .pte and tokenizer there."
                }

                val prompt = buildPrompt(confirmedCrop, predictedCondition, diagnosisAccepted)
                val answer = StringBuilder()
                var nativeError: String? = null
                val started = System.nanoTime()

                LlmModule(modelFile.absolutePath, tokenizerFile.absolutePath, 0.2f).use { llm ->
                    // Explicit load is optional, but keeps loading separate from generation.
                    // The released APIs differ in return type across versions; just call it.
                    llm.load()
                    Log.i(TAG, "Model loaded, starting offline generation")

                    val options = LlmGenerationConfig.create()
                        .seqLen(512)
                        .maxNewTokens(96)
                        .temperature(0.2f)
                        .echo(false)
                        .build()

                    llm.generate(prompt, options, object : LlmCallback {
                        override fun onResult(result: String) {
                            answer.append(result)
                            ui.post { onToken(result) }
                        }

                        override fun onStats(stats: String) {
                            Log.i(TAG, "ExecuTorch stats: $stats")
                        }

                        override fun onError(errorCode: Int, message: String) {
                            nativeError = "ExecuTorch error $errorCode: $message"
                            Log.e(TAG, nativeError!!)
                        }
                    })
                }

                nativeError?.let { throw IllegalStateException(it) }
                val elapsedMs = (System.nanoTime() - started) / 1e6
                Log.i(TAG, "Advisory completed in %.1f ms; chars=%d".format(elapsedMs, answer.length))
                ui.post { onComplete(answer.toString()) }
            } catch (e: Exception) {
                Log.e(TAG, "Offline advisory failed", e)
                ui.post { onError(e.message ?: e.javaClass.simpleName) }
            } finally {
                busy.set(false)
            }
        }
    }

    private fun buildPrompt(crop: String, predicted: String?, accepted: Boolean): String {
        val safeCondition = if (accepted && !predicted.isNullOrBlank()) {
            "The image classifier tentatively suggests '$predicted', but this is not confirmed."
        } else {
            "The image classifier could not identify a disease confidently. No diagnosis is available."
        }
        val system = """
            You provide cautious, general crop-care guidance for a farmer.
            You are not a plant pathologist and cannot verify disease from text.
            Never assert an unconfirmed diagnosis. Do not invent pesticides,
            dosages, mixtures, or chemical treatments. Do not recommend spraying.
            Provide exactly three short, low-risk next steps, such as clearer
            photographs, inspecting other leaves, and contacting local extension
            services if symptoms persist. Avoid certainty and stay under 80 words.
        """.trimIndent()
        val user = "Confirmed crop: $crop. $safeCondition Provide three safe next steps."
        return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n" +
            system + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n" +
            user + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    }

    override fun close() {
        closed.set(true)
        // Each request owns/closes its native LlmModule when generation ends.
        worker.shutdown()
    }
}
