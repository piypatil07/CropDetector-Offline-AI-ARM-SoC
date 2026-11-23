# Models, datasets, and experiment scope

## Deployed models

| Component | Inputs | Model variant | Purpose |
|---|---|---|---|
| Stage A | 160 × 160 × 3 RGB | MobileNetV2-0.35 **INT8**, 653,352 B | Plant-like vs non-plant screening |
| Stage C | 224 × 224 × 3 RGB | MobileNetV2 **INT8**, 2,755,824 B; optional FP16 and FP32 variants | 15-class supported-crop leaf classifier |
| Stage D | Text prompt | Llama 3.2 1B Instruct SpinQuant INT4, `.pte`, external | Experimental offline wording of conservative advice |

Stage A uses the exported quantized scalar mapping in its `stageA_deployment_config.json` and a provisional acceptance threshold of **0.5 (strict greater-than)**. Stage C uses calibrated output scores and the configured confidence thresholds (Automatic **0.70**, Crop-Aware **0.65**); acceptance remains tentative, not ground-truth classification.

The **15 exact class names** are defined in `android/app/src/main/assets/multiclass/model_config.json`. The readable crop/condition summary is in root README. `CropClassifier.kt` is authoritative for runtime preprocessing and calibration.

## Research data

The project experimentation uses **PlantVillage** and **PlantDoc**, with independent validation studies and non-plant photographs; provenance and split logic are described in the provided Phase 9 Colab notebooks. Original datasets are **not redistributed** here. Anyone rerunning the notebooks must independently obtain the data under the applicable terms and reproduce manifests/splits.

### Stage B not deployed

A five-class supported-crop recognizer (Tomato / Potato / Maize / Grape / Other Crop) was investigated twice. Its held-out species rejection remained inadequate, so Stage B was **not included in Android**. See [`results/research/`](../results/research/) and `docs/RESULTS_AND_LIMITATIONS.md`.

## Research notebooks included

The supplied Phase 9 `.ipynb` files cover dataset auditing, non-plant augmentation, Stage A training, external stress tests and Stage A INT8 quantization, followed by Stage B training/domain-balancing experiments. The notebook collection is not a single one-click training pipeline: most stages expect outputs from earlier runs/archives or source datasets. Read markdown at the top of each notebook and honor data licensing.

**Missing:** the original complete Stage C 15-class classifier training/export notebook was not provided with the Android source. Do not claim fully reproducible retraining of that model. The delivered binaries, JSON metadata, and golden test tensors *do* enable model-integrity and numerical consistency checks.
