# Third-party provenance and publication checklist

**Action required before making this repository Public.** Publication and redistribution permissions for the **bundled trained TFLite model weights**, pretrained architecture/backbone, sample-derived assets, and datasets must be checked by the project owner. A public GitHub repository is **not automatically open-source**. No project `LICENSE` is intentionally bundled because the owner has not chosen one.

| Component | Source / rights review to complete |
|---|---|
| Llama 3.2 1B Instruct weights/tokenizer | Meta Llama license: https://www.llama.com/llama3_2/license/ ; model distribution: https://huggingface.co/executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET ; files are **not** redistributed here |
| ExecuTorch Android | https://github.com/pytorch/executorch ; confirm version 1.5.1 and attribution/notice obligations |
| LiteRT / TensorFlow Lite | https://ai.google.dev/edge/litert ; confirm package license and notices |
| Android Jetpack CameraX, AppCompat and other Gradle dependencies | Respective open-source package licenses; dependencies fetched by Gradle |
| MobileNetV2 pretrained backbone | Establish original pretrained weight source and its exact reuse/distribution terms; do not assume due to architecture name alone |
| PlantVillage / PlantDoc | Identify exact dataset mirrors/versions, licenses, citations and any restriction on redistribution; **raw datasets not shipped** |
| Deployed Stage A and Stage C `.tflite` | Derived model redistribution rights must be approved before enabling public access |

Recommended next actions:
1. Track each third-party model, dataset and package version, source URL and relevant license in this file.
2. Obtain mentor/project-owner approval for public redistribution of the four bundled vision-model files.
3. Choose a license for the **original project code** (for example, MIT or Apache-2.0 if compatible with all dependencies), add the proper full LICENSE text only after you decide, and keep model/dataset terms distinct.
4. Review the entire public repository for personal metadata, secrets and private training samples.
5. If you cannot establish permission to redistribute vision weights, **remove the affected `.tflite` files before publishing** and offer a lawful model-download route; note this will prevent a fresh checkout from running vision inference until they are installed.

This checklist is not a legal opinion and does not assert that third-party licenses have been satisfied.
