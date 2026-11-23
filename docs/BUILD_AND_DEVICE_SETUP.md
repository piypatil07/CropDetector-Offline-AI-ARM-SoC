# Build, device, and offline model setup

## Prerequisites

- Android Studio and Android SDK platform 34, Android Debug Bridge (`adb`), USB debugging enabled on the phone.
- JDK 17 is declared by `android/gradle/gradle-daemon-jvm.properties`.
- `android/build.gradle.kts` declares Android Gradle Plugin **8.3.0** and Kotlin plugin **1.9.22**.
- **Compatibility caveat:** the uploaded `android/gradle/wrapper/gradle-wrapper.properties` points to Gradle **9.6.0**, potentially outside supported AGP 8.3.0 combinations. The uploaded app ran on the developer's machine, but a clean-checkout build with this combination has not been verified here. Use Android Studio's compatible toolchain guidance if it fails; record any adjustments (and avoid silently changing the tested runtime code).

## Build and install

```bash
cd android
./gradlew assembleDebug --no-configuration-cache
adb devices
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

On macOS, if necessary: `chmod +x gradlew`. Grant camera permission after launch.

## Get the Llama model legally (separate download)

The model is **not included** in this public repository or in the Android APK. It was obtained as an ExecuTorch-compatible pretrained export:

- Model distribution: https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET
- Exported model filename: `Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte` (previously tested; **1,135,951,488 bytes**)
- Matching tokenizer filename: `tokenizer.model` (previously tested; **2,183,982 bytes**)
- Review the Llama 3.2 license: https://www.llama.com/llama3_2/license/

Download those files in accordance with the upstream terms, and **verify their checksums against the publishing source** before deployment. The current Android code validates only the files' presence and approximate lengths, not cryptographic integrity. Do not place any access tokens in this public repository.

Make sure the device has sufficient space for the model plus runtime/cache (the test device previously had tight storage). Install the app first; then, from the directory containing both downloads:

```bash
adb shell mkdir -p /sdcard/Android/data/com.example.cropdetector/files/models
adb push Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte /sdcard/Android/data/com.example.cropdetector/files/models/
adb push tokenizer.model /sdcard/Android/data/com.example.cropdetector/files/models/
adb shell ls -lh /sdcard/Android/data/com.example.cropdetector/files/models/
```

This is **not** the LlamaDemo test path `/data/local/tmp/llama/` and is not inside the APK. In the source see `OfflineLlamaAdvisor.modelDirectory()`.

## Test procedure

1. For offline demonstrations, show Airplane mode **ON** and Wi-Fi **OFF** (a separate Wi-Fi toggle may remain enabled in Airplane mode).
2. Open the app. Wait for Stage A and Stage C numerical reference checks to complete.
3. Point at a non-plant object and press Scan: it should be rejected before Stage C. **One demonstration is not an accuracy metric.**
4. Point at a known supported crop; select the actual crop (Maize/Tomato/Potato/Grape). Inspect confidence and whether the app abstained.
5. Tap Offline AI Advisory, manually confirm crop, and capture Llama output plus the cautionary disclaimer. Keep the phone cool for performance measurements, and disclose thermal status.
6. Record ADB logs if needed:

```bash
adb logcat -d -s PlantParity CropParity PlantGate CropInference CropLlama CropLlamaUI ExecuTorch AndroidRuntime
adb shell dumpsys meminfo com.example.cropdetector
adb shell dumpsys thermalservice
```

No cloud requests are made by application code; model download and ADB file transfer necessarily occur during **preparation**, not during disconnected inference.
