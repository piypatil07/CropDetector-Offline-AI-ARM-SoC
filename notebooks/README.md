# Colab research notebooks (provided artifacts)

The `stage_a_plant_validation/` directory contains the supplied Stage A dataset, training, independent test and INT8 quantization notebooks. `stage_b_crop_recognition_experimental/` contains the two iterations of **undeployed** crop-recognition research, including two alternative notebooks for dataset corrections.

Notebooks were copied from files supplied during development; cells were **not rerun** during GitHub packaging. They contain source and markdown, not full source datasets or all previously generated checkpoint/result ZIPs. Run them in phase order, obtain the referenced PlantDoc/PlantVillage datasets under their respective licenses and provide intermediate results where instructed.

The **original Stage C 15-class disease model training notebook is absent** from the available source materials. The deployed `.tflite` binaries, config and golden tensors are under `android/`. Do not infer a training pipeline from a deployed binary alone.
