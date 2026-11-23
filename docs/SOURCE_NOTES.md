# Source audit notes

This repository was assembled from `CropDetector_Source.zip` provided by the project owner. **Active Kotlin code and its deployed models were copied without behavior changes.** Unused legacy `crop_disease_model.tflite`, local.properties, Finder artifacts, and a `.backup` copy were excluded.

- The config `multiclass/model_config.json` still has `input_validation_enabled=false`; this is a **stale metadata field**. `MainActivity.kt` actually invokes Stage A before Stage C. Config was left intact to preserve verified deployed assets.
- Stage A JSON contains a reference to an FP32 model used during development; the Android deployment only bundles the actively used INT8 model. This is expected; missing FP32 should not cause runtime failure.
- The app's `Automatic` disease mode does not implement the unshipped Stage B species-recognition gate. All public documentation must distinguish these.
- Gradle wrapper 9.6.0 and declared Android Gradle Plugin 8.3.0 may be an incompatible combination in a pristine setup. The build had succeeded earlier in the original development environment, but this new layout is only **statically checked** by this package.
- Existing `app/src/test` and `app/src/androidTest` are the Android Studio example tests; **golden tensor checks run at app initialization**, not inside an automated CI runner.
- The Llama PTE and tokenizer are intentionally not included; the Android source loads them from app-specific external files storage.
- The original Stage C training and export source was not in the provided materials and is **not recreated or falsely represented**.
