# Measurements, evidence and scientific limitations

## Test levels

1. **Static binary integrity:** hashes and sizes of four bundled deployed vision model files match their deployment JSON files.
2. **Numerical golden parity:** Stage A 8/8 and Stage C 12/12 checks previously passed on the user's device. These establish run-time numeric consistency for given tensors, not real-world accuracy.
3. **Camera smoke tests:** Nine scans in uploaded 23 Nov logs. Stage A accepted 8 and rejected 1. Stage C received 8 images, accepted 2 tentative outputs, abstained on 6. No actual object/crop labels were recorded in test notes; **cannot calculate field accuracy** from this run.
4. **LLM integration:** Two maize advisories completed in the same log capture. This confirms local inference and UI flow, but no full advisory text or expert agronomic grading was in these test files.

## 23 November repeat session (connectivity on)

| Run | Crop | Disease condition passed to Llama? | Prompt tokens | Generated tokens | Prompt eval | Decode speed | LLM inference | App elapsed |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Maize | No — classifier abstained | 136 | 88 | 7.468 s | 8.934 tok/s | 17.318 s | 23.874 s |
| 2 | Maize | Yes — tentative accepted condition | 141 | 137 | 8.767 s | 5.521 tok/s | 33.581 s | 42.182 s |

**Important:** captured device settings were Airplane mode **OFF**, Wi-Fi **ON**, and a later thermal-service snapshot recorded status **3** (severe). The app does not require the network at inference, but this particular run **must not be labelled disconnected/offline proof**, nor an unthrottled benchmark. Full data in `results/2026-11-23`.

## Earlier separately reported disconnected session

The user reported Airplane mode ON and Wi-Fi OFF while getting a Llama advisory; logging showed 138 prompt tokens, 97 generated tokens, 9.3575 decode tokens/sec, 17.843 s inference and 24.691 s application elapsed, ~901 MiB RSS at end of generation. Connectivity was **user-confirmed**, not independently recorded by a separate network monitor. These figures must remain separate from the thermally challenged later session.

## Stage B comparison (experimental, no deployment)

| Experiment | In-domain metric | Unsupported species rejection | Status |
|---|---|---|---|
| B1: PlantDoc Stage B | 75.28% development accuracy (N=360) | 59.41% (N=303) | Not deployed |
| B2: PlantDoc + PlantVillage | 72.78% PlantDoc development accuracy (N=360) | 53.47% retrospective unseen species (N=303) | Not deployed |

B2 PlantVillage development accuracy was 95.55% (N=517), illustrating domain sensitivity; it must not be advertised as external field accuracy. See `results/research/*.json` for the original recorded experimental metrics.

## Major open limitations

- Stage A is not a plant-species verifier; unsupported crop leaves can pass.
- Stage B is not deployed. `Automatic` classification must not be equated with validated crop recognition.
- Stage C could predict a disease on a non-target crop and confidence may be poorly calibrated on field images.
- Stage C's original full training/export notebook is missing from the provided archive.
- LLM output is unverified and may be generic, overconfident or impractical offline; no expert-approved grounding library is shipped.
- Data use and redistribution rights remain to be audited before public distribution.
- Fresh-checkout Android build and cold-device, repeated thermal-controlled benchmarks are pending.
