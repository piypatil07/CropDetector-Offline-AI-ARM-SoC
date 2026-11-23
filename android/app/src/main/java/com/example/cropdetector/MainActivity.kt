package com.example.cropdetector

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import org.json.JSONObject

class MainActivity : AppCompatActivity() {
    private lateinit var previewView: PreviewView
    private lateinit var scanButton: Button
    private lateinit var advisoryButton: Button
    private lateinit var resultText: TextView
    private var imageCapture: ImageCapture? = null
    private val worker = Executors.newSingleThreadExecutor()
    // Only the worker thread uses or closes the interpreter.
    private var classifier: CropClassifier? = null
    // Stage A general plant-like validator. Shares the same worker as disease inference.
    private var plantValidator: PlantValidator? = null
    private var offlineAdvisor: OfflineLlamaAdvisor? = null

    // Only set after Stage A accepts and Stage C has returned a prediction.
    // Access this record on the UI thread, never from the inference worker.
    private data class LastScan(
        val selectedCrop: String,
        val topClassName: String,
        val diseaseAccepted: Boolean
    )
    private var lastScan: LastScan? = null
    private var scanSummary: String? = null
    // UI state is accessed on the main thread only.
    private var modelReady = false
    private var busy = false
    private var cropOptions = listOf(CropClassifier.AUTO)
    private var precision = "INT8"
    private var threads = 2
    private var passedFormats = emptyList<String>()
    private lateinit var config: JSONObject // written and read on worker only
    private val destroyed = AtomicBoolean(false)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.previewView)
        scanButton = findViewById(R.id.scanButton)
        advisoryButton = findViewById(R.id.advisoryButton)
        resultText = findViewById(R.id.resultText)
        scanButton.isEnabled = false
        advisoryButton.isEnabled = false
        offlineAdvisor = OfflineLlamaAdvisor(applicationContext)
        advisoryButton.setOnClickListener { chooseAdvisoryCrop() }
        resultText.text = "Checking multiclass models…"
        scanButton.setOnClickListener { chooseCrop() }
        scanButton.setOnLongClickListener {
            if (modelReady && !busy) showEngineeringMenu()
            true
        }

        loadModels()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            setupCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 100)
        }
    }

    private fun updateScanButton() {
        scanButton.isEnabled = modelReady && imageCapture != null && !busy
        advisoryButton.isEnabled = modelReady && lastScan != null && !busy
    }

    private fun loadModels() {
        worker.execute {
            try {
                config = CropClassifier.readConfig(assets)
                val results = CropClassifier.runParityTests(assets, config)
                Log.i("CropParity", "All results: $results")
                // Require passing parity for the selected INT8 disease model; no silent fallback.
                require(results[precision] == true) { "$precision parity needs review. See CropParity logs." }
                // Validate Stage A on exact Colab tensors before enabling camera diagnosis.
                require(PlantValidator.runParityTests(assets)) {
                    "Stage A numerical parity failed; see PlantParity logs"
                }
                plantValidator = PlantValidator(assets, threads = 2)
                val loaded = CropClassifier(assets, config, precision, threads)
                classifier = loaded
                val options = loaded.cropOptions
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    passedFormats = results.filterValues { it }.keys.toList()
                    cropOptions = options
                    modelReady = true
                    resultText.text = "$precision disease model + Stage A plant gate ready. Tap Scan to select crop; long-press Scan for engineering settings.\nExperimental: supported-crop verification is not yet implemented."
                    updateScanButton()
                }
            } catch (e: Exception) {
                // Fail closed: do not enable camera diagnosis without BOTH models.
                classifier?.close()
                classifier = null
                plantValidator?.close()
                plantValidator = null
                Log.e("CropModel", "Multiclass initialization failed", e)
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    resultText.text = "Model setup failed: ${e.message}"
                    updateScanButton()
                }
            }
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 100) {
            if (grantResults.firstOrNull() == PackageManager.PERMISSION_GRANTED) setupCamera()
            else resultText.text = "Camera permission is required for scanning."
        }
    }

    private fun setupCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            if (!isDestroyed && !isFinishing) {
                try {
                    val provider = future.get()
                    val preview = Preview.Builder().build().also {
                        it.setSurfaceProvider(previewView.surfaceProvider)
                    }
                    val capture = ImageCapture.Builder().build()
                    provider.unbindAll()
                    provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                    imageCapture = capture
                    updateScanButton()
                } catch (e: Exception) {
                    imageCapture = null
                    updateScanButton()
                    Log.e("CropCamera", "Camera bind failed", e)
                    resultText.text = "Camera setup failed: ${e.message}"
                }
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun showEngineeringMenu() {
        AlertDialog.Builder(this)
            .setTitle("Engineering mode — $precision / $threads threads")
            .setItems(arrayOf("Choose model precision", "Choose CPU threads", "Run CPU benchmark")) { _, index ->
                when (index) {
                    0 -> AlertDialog.Builder(this).setTitle("Model precision")
                        .setItems(passedFormats.toTypedArray()) { _, i -> switchModel(passedFormats[i], threads) }
                        .setNegativeButton("Cancel", null).show()
                    1 -> AlertDialog.Builder(this).setTitle("CPU threads")
                        .setItems(arrayOf("1", "2", "4")) { _, i -> switchModel(precision, listOf(1,2,4)[i]) }
                        .setNegativeButton("Cancel", null).show()
                    2 -> AlertDialog.Builder(this).setTitle("Run benchmark?")
                        .setMessage("Runs 3 rounds, 40 measurements per configuration. Keep the app open. It pauses to cool at thermal status 2 or higher before each trial and stops if severe throttling is detected. Optimized benchmark CSV files will be replaced; original benchmark files are kept.")
                        .setPositiveButton("Run") { _, _ -> runBenchmark() }
                        .setNegativeButton("Cancel", null).show()
                }
            }.setNegativeButton("Cancel", null).show()
    }

    private fun switchModel(format: String, count: Int) {
        if (busy || !modelReady || (format == precision && count == threads)) return
        val oldFormat = precision
        val oldCount = threads
        busy = true
        updateScanButton()
        resultText.text = "Loading $format / $count threads…"
        worker.execute {
            try {
                classifier?.close()
                classifier = null
                classifier = CropClassifier(assets, config, format, count)
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    precision = format
                    threads = count
                    showResult("$format / $count threads ready. Tap Scan or long-press for engineering settings.")
                }
            } catch (e: Exception) {
                Log.e("CropModel", "Switch failed", e)
                try {
                    classifier = CropClassifier(assets, config, oldFormat, oldCount)
                    showResult("Switch failed; restored $oldFormat / $oldCount threads. ${e.message}")
                } catch (restore: Exception) {
                    Log.e("CropModel", "Restore failed", restore)
                    runOnUiThread {
                        if (isDestroyed || isFinishing) return@runOnUiThread
                        modelReady = false
                        showResult("Model restore failed; restart the app. ${restore.message}")
                    }
                }
            }
        }
    }

    private fun runBenchmark() {
        if (busy || !modelReady) return
        val format = precision
        val count = threads
        val formats = passedFormats.toList()
        busy = true
        updateScanButton()
        resultText.text = "Preparing benchmark…"
        // Prevent screen sleep; keep camera preview active as in normal app usage.
        window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        worker.execute {
            var message: String
            try {
                classifier?.close()
                classifier = null
                // Keep the original disease-only benchmark's RAM measurements comparable.
                plantValidator?.close()
                plantValidator = null
                message = CropBenchmark.run(this, config, formats, { destroyed.get() }) { status ->
                    runOnUiThread {
                        if (!isDestroyed && !isFinishing) resultText.text = status
                    }
                }
            } catch (e: Exception) {
                Log.e("CropBenchmark", "Benchmark stopped", e)
                message = "Benchmark stopped: ${e.message}. Completed rows may be saved."
            }
            if (!destroyed.get()) {
                var restored = false
                try {
                    plantValidator = PlantValidator(assets, threads = 2)
                    classifier = CropClassifier(assets, config, format, count)
                    restored = true
                } catch (e: Exception) {
                    classifier?.close()
                    classifier = null
                    plantValidator?.close()
                    plantValidator = null
                    Log.e("CropModel", "Restore after benchmark failed", e)
                    message += "\nModel restore failed; restart app."
                }
                val finalMessage = message
                runOnUiThread {
                    if (isDestroyed || isFinishing) return@runOnUiThread
                    window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
                    modelReady = restored
                    showResult(finalMessage)
                }
            }
        }
    }

    private fun chooseCrop() {
        AlertDialog.Builder(this)
            .setTitle("Select crop")
            .setItems(cropOptions.toTypedArray()) { _, i -> takePhotoAndAnalyze(cropOptions[i]) }
            .setNegativeButton("Cancel", null)
            .show()
    }

    /** A crop confirmation is mandatory even if the scan used Automatic mode. */
    private fun chooseAdvisoryCrop() {
        if (busy || !modelReady) return
        val scan = lastScan ?: return

        val crops = arrayOf("Tomato", "Potato", "Maize", "Grape")

        AlertDialog.Builder(this)
            .setTitle("Confirm the photographed crop")
            .setItems(crops) { _, which ->
                startOfflineAdvice(scan, crops[which])
            }
            .setNegativeButton("Cancel", null)
            .show()
    }

    private fun startOfflineAdvice(scan: LastScan, confirmedCrop: String) {
        if (busy || !modelReady || lastScan !== scan) return
        val engine = offlineAdvisor ?: return
        if (!engine.modelFilesPresent()) {
            AlertDialog.Builder(this)
                .setTitle("Offline Llama model not found")
                .setMessage("Copy the verified Llama .pte and tokenizer.model to the CropDetector app-specific files/models directory. See the integration README. No network download is attempted.")
                .setPositiveButton("OK", null)
                .show()
            return
        }

        // Only pass a tentative condition if Stage C accepted it AND the
        // verified crop agrees with both the selection and the output label.
        val matchesLabel = scan.topClassName.startsWith("$confirmedCrop - ")
        val matchesSelection = scan.selectedCrop == confirmedCrop ||
            scan.selectedCrop == CropClassifier.AUTO
        val passCondition = scan.diseaseAccepted && matchesLabel && matchesSelection
        val tentativeCondition = if (passCondition) scan.topClassName else null
        val previousResult = scanSummary ?: resultText.text.toString()
        val rendered = StringBuilder()

        busy = true // Do not run TFLite benchmarks/scans while the LLM has a large memory footprint.
        updateScanButton()
        resultText.text = previousResult + "\n\nLoading offline Llama for $confirmedCrop…"
        Log.i("CropLlamaUI", "Request crop=$confirmedCrop tentativeConditionPassed=$passCondition")

        engine.generate(
            confirmedCrop = confirmedCrop,
            predictedCondition = tentativeCondition,
            diagnosisAccepted = passCondition,
            onToken = { token ->
                if (!isDestroyed && !isFinishing) {
                    rendered.append(token)
                    resultText.text = previousResult +
                        "\n\nOffline AI guidance (experimental):\n" + rendered.toString()
                }
            },
            onComplete = { answer ->
                if (!isDestroyed && !isFinishing) {
                    val finalAnswer = answer.ifBlank { rendered.toString() }.trim()
                    val display = if (finalAnswer.isBlank()) {
                        "No text was generated. Please retry the advisory."
                    } else {
                        "Offline AI guidance (experimental):\n$finalAnswer\n\n" +
                            "Generated advice is unverified. Consult a local agricultural expert before treatment."
                    }
                    showResult(previousResult + "\n\n" + display, scan)
                }
            },
            onError = { message ->
                Log.e("CropLlamaUI", "Advisor failed: $message")
                if (!isDestroyed && !isFinishing) {
                    showResult(previousResult + "\n\nOffline advisor unavailable: $message", scan)
                }
            }
        )
    }

    private fun takePhotoAndAnalyze(crop: String) {
        val capture = imageCapture ?: return
        if (!modelReady || busy) return
        busy = true
        lastScan = null // Old predictions must never be used for a new photograph.
        scanSummary = null
        updateScanButton()
        resultText.text = "Analyzing…"
        previewView.display?.let { capture.targetRotation = it.rotation }
        capture.takePicture(ContextCompat.getMainExecutor(this), object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                if (isDestroyed || isFinishing) {
                    image.close()
                    return
                }
                val bitmap: Bitmap
                val rotation: Int
                try {
                    rotation = image.imageInfo.rotationDegrees
                    bitmap = image.toBitmap()
                } catch (e: Exception) {
                    showResult("Image conversion failed: ${e.message}")
                    return
                } finally {
                    image.close()
                }
                worker.execute {
                    var rotated: Bitmap? = null
                    try {
                        val upright = if (rotation == 0) bitmap else Bitmap.createBitmap(
                            bitmap, 0, 0, bitmap.width, bitmap.height,
                            Matrix().apply { postRotate(rotation.toFloat()) }, true
                        )
                        rotated = upright
                        val gate = checkNotNull(plantValidator) {
                            "Stage A validator is unavailable; diagnosis blocked"
                        }
                        val check = gate.validate(upright)
                        Log.i("PlantGate", String.format(Locale.US,
                            "plantScore=%.6f threshold=%.4f accepted=%s nonplantLogit=%.4f " +
                                "prepMs=%.3f runMs=%.3f totalMs=%.3f",
                            check.plantScore, check.threshold, check.accepted,
                            check.nonplantLogit, check.preparationMs,
                            check.interpreterMs, check.totalMs))
                        if (!check.accepted) {
                            // No disease inference for images rejected by Stage A.
                            showResult(buildString {
                                append("No plant-like subject detected. No diagnosis.\n")
                                append("Plant-like score: ${percent(check.plantScore)}\n")
                                append(String.format(Locale.US,
                                    "Stage A interpreter: %.2f ms\n", check.interpreterMs))
                                append("Try another clear leaf photograph. This is an experimental filter; it can reject real plants.")
                            })
                        } else {
                            val model = checkNotNull(classifier)
                            val prediction = model.predict(upright, crop)
                            val best = prediction.candidates.first()
                            val text = buildString {
                                append("${model.precision} | $crop\n")
                                append("Plant-like score: ${percent(check.plantScore)}\n")
                                if (prediction.accepted) {
                                    append("Possible condition: ${best.name}\n")
                                    append("Model score: ${percent(best.score)}\n")
                                } else {
                                    append("Uncertain disease result — no diagnosis.\n")
                                    append("Highest disease score: ${percent(best.score)}\n")
                                }
                                append("Disease threshold: ${percent(prediction.threshold)}\n")
                                if (prediction.accepted) {
                                    for (candidate in prediction.candidates.drop(1)) {
                                        append("Alternative: ${candidate.name} (${percent(candidate.score)})\n")
                                    }
                                }
                                append(String.format(Locale.US,
                                    "Stage A run: %.2f ms | Disease run: %.2f ms\n",
                                    check.interpreterMs, prediction.latencyMs))
                                append("Prototype only. Plant-like does not mean supported crop; confirm independently.")
                            }
                            Log.i("CropInference", "${model.precision} crop=$crop top=${best.name} " +
                                "score=${best.score} accepted=${prediction.accepted} runMs=${prediction.latencyMs}")
                            showResult(text, LastScan(crop, best.name, prediction.accepted))
                        }
                    } catch (e: Exception) {
                        Log.e("CropInference", "Inference failed", e)
                        showResult("Inference failed: ${e.message}")
                    } finally {
                        rotated?.let { if (it !== bitmap) it.recycle() }
                        bitmap.recycle()
                    }
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e("CropCamera", "Capture failed", exception)
                showResult("Error capturing image: ${exception.message}")
            }
        })
    }

    private fun percent(score: Double): String = String.format(Locale.US, "%.1f%%", score * 100)

    private fun showResult(message: String, scan: LastScan? = null) {
        runOnUiThread {
            if (isDestroyed || isFinishing) return@runOnUiThread
            resultText.text = message
            if (scan == null) {
                scanSummary = null
            } else if (lastScan !== scan) {
                scanSummary = message // Do not stack earlier Llama output on repeated requests.
            }
            lastScan = scan
            busy = false
            updateScanButton()
        }
    }

    override fun onDestroy() {
        destroyed.set(true)
        offlineAdvisor?.close() // Executor shutdown; any in-flight generation finishes on its worker.
        offlineAdvisor = null
        // Serialize cleanup after any in-flight model work.
        worker.execute {
            classifier?.close()
            classifier = null
            plantValidator?.close()
            plantValidator = null
        }
        worker.shutdown()
        super.onDestroy()
    }
}
