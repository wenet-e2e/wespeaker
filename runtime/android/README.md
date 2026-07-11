# WeSpeaker Android Speaker Verification Demo

This app extracts speaker embeddings from two **16 kHz PCM WAV** clips on device, computes cosine similarity (same mapping to 0–1 as desktop `runtime/onnxruntime` `asv_main`), and compares against a threshold to decide same vs different speaker.

## ONNX model

Export ONNX on a PC following the repo docs:

```bash
python wespeaker/bin/export_onnx.py \
  --config $exp/config.yaml \
  --checkpoint $exp/avg_model.pt \
  --output_model final.onnx
```

Copy `final.onnx` to `app/src/main/assets/final.onnx` and build (filename must match).

## Build

**JDK 17 or newer** is required (Android Gradle Plugin 8.x). If only Java 8 is installed, install JDK 17 or pick the bundled JDK under Android Studio *Settings → Build → Gradle → Gradle JDK*.

Open `runtime/android` in Android Studio, or use the Gradle wrapper:

```bash
cd runtime/android
./gradlew :app:assembleDebug
```

The app depends on [ONNX Runtime Android](https://github.com/microsoft/onnxruntime) (`onnxruntime-android` AAR). Native integration matches [wekws/runtime/android](https://github.com/wenet-e2e/wekws/tree/main/runtime/android): a resolvable `extractForNativeBuild` configuration unpacks `headers/` and `jni/` from the AAR; CMake uses `include_directories` and links `libonnxruntime.so` (no Prefab / `find_package`). App logic reuses this repo’s `runtime/core` Fbank, `SpeakerEngine`, and ONNX backend.

## Usage

1. After installing the APK, pick enroll and test WAV files (16 kHz recommended; other rates are not resampled in-app and may hurt quality).
2. Tune threshold, embedding dim, and chunk samples to match training/export settings.
3. Tap **Compare** to see similarity score and same/different verdict.

## Notes

- If CMake reports missing `onnxruntime*.aar` extract dir, **Sync** and run a full **Build** so `extractAARForNativeBuild` runs before `configureCMake` (same idea as wekws).
- First inference copies the model from assets to app-private storage; ensure `assets/final.onnx` exists and is non-empty.
- Default threshold is 0.5; tune on your validation set.
