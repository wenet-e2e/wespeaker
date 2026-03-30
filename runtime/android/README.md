# WeSpeaker Android 说话人验证 Demo

类似 [WeNet Android Demo](https://github.com/wenet-e2e/wenet/tree/main/runtime/android)，本工程在设备端对两段 **16 kHz PCM WAV** 提取说话人向量并计算余弦相似度（与桌面端 `runtime/onnxruntime` 的 `asv_main` 一致，得分映射到 0–1），再与阈值比较判断是否同一人。

## 准备 ONNX 模型

在 PC 上按仓库文档导出 ONNX：

```bash
python wespeaker/bin/export_onnx.py \
  --config $exp/config.yaml \
  --checkpoint $exp/avg_model.pt \
  --output_model final.onnx
```

将 `final.onnx` 复制到 `app/src/main/assets/final.onnx` 后编译（文件名需一致）。

参数需与模型一致，例如常见 **embedding 维度 256**；`samples_per_chunk` 与 `extract_emb_main` / `asv_main` 相同（`-1` 表示整段音频一个 embedding）。

## 编译

**需要 JDK 17 或以上**（Android Gradle Plugin 8.x 要求；仅系统自带 Java 8 时请先安装 JDK 17 或在 Android Studio 的 *Settings → Build → Gradle → Gradle JDK* 中选择内置 JDK）。

用 Android Studio 打开本目录 `runtime/android`，或使用 Gradle Wrapper：

```bash
cd runtime/android
./gradlew :app:assembleDebug
```

依赖 [ONNX Runtime Android](https://github.com/microsoft/onnxruntime)（`onnxruntime-android` AAR）。Native 集成方式与 [wekws/runtime/android](https://github.com/wenet-e2e/wekws/tree/main/runtime/android) 相同：可解析配置 `extractForNativeBuild` 解压 AAR 中的 `headers/`、`jni/`，CMake 直接 `include_directories` + 链接 `libonnxruntime.so`（不依赖 Prefab/`find_package`）。业务代码仍复用本仓库 `runtime/core` 的 Fbank、`SpeakerEngine` 与 ONNX 后端。

## 使用

1. 安装 APK 后，依次选择「注册」「测试」两段 WAV（建议 16 kHz；其它采样率未做重采样，可能影响效果）。
2. 调整阈值与 embedding 维度、chunk 采样数（与训练/导出配置一致）。
3. 点击「比对」查看相似度得分与是否同一人。

## 说明

- 若 CMake 报未找到 `onnxruntime*.aar` 解压目录，请先 **Sync** 并执行一次完整 **Build**，确保 `extractAARForNativeBuild` 在 `configureCMake` 之前跑完（与 wekws 的 `extractAARForNativeBuild` 同理）。
- 首推理会从 assets 拷贝模型到应用私有目录，需保证 `assets/final.onnx` 存在且非空。
- 阈值默认 0.5，可按验证集自行标定。
