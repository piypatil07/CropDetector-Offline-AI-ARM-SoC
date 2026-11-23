# Android inference architecture

## Source of truth

The active Android code lives in `android/app/src/main/java/com/example/cropdetector/`:

- `MainActivity.kt`: CameraX capture, UI selections, Stage A gate, Stage C invocation, engineering benchmarks, and Llama advisory button.
- `PlantValidator.kt`: Stage A INT8 plant-like screening; acceptance is **plantScore > 0.5**, not supported-crop certification.
- `CropClassifier.kt`: Stage C 15-class classification, input preprocessing, model precision selection, calibration and confidence abstention.
- `CropBenchmark.kt`: CPU timing/thermal performance measurements for disease-model precisions/threads.
- `OfflineLlamaAdvisor.kt`: background on-device ExecuTorch model loading, token streaming, and local prompt construction.

## High-level execution

1. On launch: run 12 Stage C golden numerical vector checks (4 per precision) plus 8 Stage A checks. A failing required INT8 check disables scanning rather than silently falling back.
2. Android CameraX captures a photo. Stage A rejects obvious non-plant images. **It can still accept unsupported crop leaves**, and its result is not a field-ground-truth label.
3. The user selects one of four crops or experimental `Automatic` mode. Stage C infers one of 15 classes. Crop-aware mode restricts eligible classes; automatic mode does not validate the actual species.
4. A threshold-based acceptance/abstention rule prevents low-confidence disease labels from being sent as accepted predictions.
5. Before advisory, a separate modal asks the user to **confirm the photographed crop**. The LLM receives a possible condition only if Stage C accepted it and the predicted class prefix is consistent with the confirmed crop and previous scan mode.
6. Llama 3.2 1B Instruct SpinQuant INT4 is loaded from app-owned storage; tokens are streamed to the UI. The generated text always carries a human-expert disclaimer. This is not validated treatment guidance.

The working deployment has **no Stage B crop recognizer**. Stage B notebooks are experimental and should not be reported as deployed functionality. There is no pest-specific detector or knowledge-grounded retrieval system.

## Runtime connectivity

The Android manifest declares `android.permission.CAMERA` but not `android.permission.INTERNET`. LLM model weights are copied beforehand; runtime generation occurs inside the Android process using ExecuTorch. A convincing demo should nevertheless visibly establish Airplane ON and Wi-Fi OFF, and state model-file setup was completed in advance.
