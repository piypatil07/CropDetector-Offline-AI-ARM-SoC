#!/usr/bin/env bash
# Run after installing the APK and establishing device permissions. Read-only evidence capture.
set -euo pipefail
OUT="${1:-device-evidence-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT"
{
  date -Iseconds
  adb shell getprop ro.product.model
  adb shell getprop ro.build.version.release
  adb shell getprop ro.product.cpu.abi
  printf 'airplane_mode (1=on): '; adb shell settings get global airplane_mode_on
  printf 'wifi_on (1=on): '; adb shell settings get global wifi_on
  adb shell df -h /data
  adb shell grep MemAvailable /proc/meminfo
} > "$OUT/preflight.txt"
adb logcat -d -s PlantParity CropParity PlantGate CropInference CropLlama CropLlamaUI ExecuTorch AndroidRuntime > "$OUT/inference.log"
adb shell dumpsys meminfo com.example.cropdetector > "$OUT/meminfo.txt"
adb shell dumpsys thermalservice > "$OUT/thermalservice.txt"
printf 'Saved %s. Review for personal details before publishing.\n' "$OUT"
