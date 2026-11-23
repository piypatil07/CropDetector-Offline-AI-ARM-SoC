# CropDetector — Offline Plant Disease Screening and AI Advisory on ARM

**ARM SoC Challenge · Android research prototype · 23 November 2026**

CropDetector is an **on-device, offline-capable Android prototype** that screens camera images for a plant-like subject, estimates a possible leaf condition for four supported crops, and generates brief experimental agricultural guidance using a locally executed 1B-parameter language model. It is a **research demonstration, not a validated diagnosis or treatment product**.

**Target tested phone:** Samsung Galaxy A14 5G (SM-A146B), Exynos 1330 (`arm64-v8a`), Android 15. The vision models use LiteRT/TFLite on CPU; the text generator uses ExecuTorch on CPU.

> **Uncertainty matters:** Stage A is a plant-*like* gate, not supported-crop recognition. Stage B automatic crop identification was researched but **not deployed** because held-out/unsupported species were too often accepted. Manually choose the actual crop for the disease scan and **confirm it again** before asking for advice. The screen also offers an experimental `Automatic` disease-classification option; do not treat it as reliable crop identification. Stage C can abstain from an uncertain diagnosis. Llama output is experimental and unverified.

## Pipeline

```text
CameraX photo
    |
    v
Stage A: plant-like / non-plant validator (MobileNetV2, INT8, 160 x 160)
    | rejected -> no diagnosis / no advisory
    v
User's crop selection: Maize / Tomato / Potato / Grape
    |  (Automatic is an experimental disease-classifier mode, not crop verification)
    v
Stage C: 15-class disease model (MobileNetV2, 224 x 224; FP32 / FP16 / INT8)
    |
    +-- low-confidence -> abstain (no condition supplied to Llama)
    +-- accepted -> tentative condition (still not ground truth)
    v
Manual confirmation of real crop before advisory
    |
    v
ExecuTorch + Llama 3.2 1B Instruct SpinQuant INT4 (on-device text generation)
    |
    v
Experimental suggestions + disclaimer
```

## Supported crop/disease classes

| Crop | Classes |
|---|---|
| Maize | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Healthy |

These are **15 classifier outputs**, not 15 validated field diagnoses. See [`docs/MODELS_AND_DATASETS.md`](docs/MODELS_AND_DATASETS.md) for the exact internal labels.

## Quick start

1. Install Android Studio, Android SDK **platform 34**, a compatible JDK/Gradle toolchain, and Android Debug Bridge (`adb`). Read the [known toolchain compatibility caveat](docs/BUILD_AND_DEVICE_SETUP.md) before building on a fresh environment.
2. Clone this repository and open [`android/`](android/) as the Android Studio project.
3. From a terminal:

   ```bash
   cd android
   ./gradlew assembleDebug --no-configuration-cache
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

4. Grant camera permission. The plant gate + disease classifier run using the TFLite models already bundled in `android/app/src/main/assets/`.
5. For **offline advice**, obtain the licensed ExecuTorch Llama `.pte` weights and matching tokenizer **separately** and transfer them into the app-specific `files/models/` directory, as described in [`docs/BUILD_AND_DEVICE_SETUP.md`](docs/BUILD_AND_DEVICE_SETUP.md). They are **not** in this repository.
6. For a deliberately disconnected test, enable Airplane mode, separately turn Wi-Fi off, then scan a known supported crop and request advice. Record device conditions for any reported performance.

**Important:** The Android source was successfully run on the development handset, but this repackaged repository has **not been independently compiled on a clean Android SDK environment**. Use the checks below before making a reproducibility claim.

## Measured progress (not field-diagnosis accuracy)

| Evidence | Observation | Meaning / limitations |
|---|---|---|
| On-device numerical parity | **20/20** golden vectors: Stage A 8/8 and Stage C 12/12 | Export/runtime consistency; not classification accuracy |
| Deployed Stage A INT8 | **653,352 bytes** (about 0.62 MiB) | 160 × 160 RGB input |
| Deployed Stage C INT8 | **2,755,824 bytes** (about 2.63 MiB) | 224 × 224 RGB, calibrated abstention |
| Earlier reported disconnected advisory | **97 generated tokens; 9.36 tokens/s decode; 24.69 s app elapsed** | User reported Airplane mode ON + Wi-Fi OFF; no independent network trace |
| 23 Nov repeat trial 1 | **88 tokens; 8.93 tokens/s; 23.87 s app elapsed** | Wi-Fi ON, Airplane OFF in this particular session |
| 23 Nov repeat trial 2 | **137 tokens; 5.52 tokens/s; 42.18 s app elapsed** | Later thermal snapshot status 3 (severe), so do not call unthrottled |
| 23 Nov camera logs | **9 scans**; Stage A rejected 1; Stage C accepted 2/8 and abstained 6/8 | True object/leaf labels not recorded, so no field accuracy conclusion |

Raw values used for the final two rows are in [`results/2026-11-23/`](results/2026-11-23/). Further quantitative caveats: [`docs/RESULTS_AND_LIMITATIONS.md`](docs/RESULTS_AND_LIMITATIONS.md).

## Repository map

| Folder/file | Purpose |
|---|---|
| [`android/`](android/) | Original Android Studio project, working Kotlin UI, CameraX, LiteRT, ExecuTorch integration, bundled TFLite weights and golden vectors |
| [`notebooks/`](notebooks/) | Stage A and experimental Stage B Colab notebooks supplied by the project; original Stage C training notebook is **missing** |
| [`results/`](results/) | Source-derived Stage B metrics and selected device benchmarking CSVs |
| [`docs/`](docs/) | Architecture, build instructions, data/model provenance, limitations and licenses/attributions to review |
| [`scripts/verify_models.py`](scripts/verify_models.py) | Recompute SHA-256 and validate bundled weight files, metadata and golden-vector dimensions |
| [`demo/`](demo/) | Two-minute video storyboard and public-video placeholder; no recording bundled |
| [`GITHUB_PUBLISH.md`](GITHUB_PUBLISH.md) | Exact steps to publish a **public** repository |

## Verification (Python 3 standard library only)

```bash
python3 scripts/verify_models.py
```

The script is a **static asset integrity check**, not a test that the classifiers run or that advice is correct. Runtime parity checks are already implemented in `MainActivity.kt` during model setup.

## Reproducibility and responsible use

- The original **15-class Stage C training/export notebook was not in the provided files**. Its deployed model, configuration, and runtime parity vectors are included, but a fresh end-to-end retraining from original data is not demonstrated by this repository.
- The failed Stage B model is an **experiment only**, not an app feature; metrics and notebooks are preserved for transparency.
- The assistant-generated advice is **not grounded in a verified agricultural knowledge base**. Never use this app alone to select chemicals, dosage, or other irreversible treatments.
- Some configuration metadata predates the Stage A app integration; see [`docs/SOURCE_NOTES.md`](docs/SOURCE_NOTES.md).
- Dataset and model redistribution permissions must be reviewed before setting the repository Public. See [`docs/ATTRIBUTION_AND_LICENSES.md`](docs/ATTRIBUTION_AND_LICENSES.md). **No project-level license is selected yet**; public visibility does not grant other users reuse rights.
- Reviewer demo video and final report will be added when finalized. Do not mistake this README for a completed challenge submission.

## Project status and challenge deliverables

- [x] Android prototype, models, device inference and numerical reference checks
- [x] Clean source layout, available research notebooks, selected benchmarking records and setup docs
- [ ] Rebuild from a fresh checkout / compatible toolchain
- [ ] Third-party licensing review and select a project license
- [ ] Add missing original Stage C training/export material if obtainable

Project evidence and limitations are described for engineering evaluation; no warranty of agronomic accuracy is made.
