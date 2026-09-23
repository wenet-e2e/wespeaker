// Copyright 2023 Chengdong Liang (WeSpeaker runtime)
// SPDX-License-Identifier: Apache-2.0

#include <jni.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "frontend/wav.h"
#include "glog/logging.h"
#include "speaker/speaker_engine.h"

namespace {

std::once_flag g_glog_init;

void EnsureGlog() {
  std::call_once(g_glog_init, []() {
    google::InitGoogleLogging("wespeaker");
    FLAGS_logtostderr = 1;
    FLAGS_minloglevel = 2;
  });
}

jfloatArray MakeFloatArray(JNIEnv* env, float a, float b) {
  jfloatArray out = env->NewFloatArray(2);
  if (!out) return nullptr;
  jfloat buf[2] = {a, b};
  env->SetFloatArrayRegion(out, 0, 2, buf);
  return out;
}

void ThrowIo(JNIEnv* env, const char* msg) {
  jclass ex = env->FindClass("java/io/IOException");
  if (ex) env->ThrowNew(ex, msg);
}

std::string JStringToUtf8(JNIEnv* env, jstring s) {
  if (!s) return {};
  const char* p = env->GetStringUTFChars(s, nullptr);
  std::string out(p ? p : "");
  if (p) env->ReleaseStringUTFChars(s, p);
  return out;
}

}  // namespace

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_wespeaker_app_WespeakerNative_compare(JNIEnv* env, jclass /* clazz */,
                                               jstring j_enroll, jstring j_test,
                                               jstring j_model,
                                               jdouble j_threshold,
                                               jint j_fbank_dim,
                                               jint j_sample_rate) {
  EnsureGlog();

  const std::string enroll_path = JStringToUtf8(env, j_enroll);
  const std::string test_path = JStringToUtf8(env, j_test);
  const std::string model_path = JStringToUtf8(env, j_model);
  const float threshold = static_cast<float>(j_threshold);

  if (enroll_path.empty() || test_path.empty() || model_path.empty()) {
    ThrowIo(env, "路径不能为空");
    return nullptr;
  }

  try {
    wenet::WavReader enroll_reader;
    if (!enroll_reader.Open(enroll_path)) {
      ThrowIo(env, "无法打开注册音频（需有效 WAV）");
      return nullptr;
    }
    wenet::WavReader test_reader;
    if (!test_reader.Open(test_path)) {
      ThrowIo(env, "无法打开测试音频（需有效 WAV）");
      return nullptr;
    }
    if (enroll_reader.num_sample() <= 0 || test_reader.num_sample() <= 0) {
      ThrowIo(env, "音频长度无效");
      return nullptr;
    }

    auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
        model_path, j_fbank_dim, j_sample_rate,
        0 /* embedding size: infer from ONNX output shape */,
        -1 /* one embedding for full audio; same as per_chunk_samples_ <= 0 */);
    const int embedding_size = speaker_engine->EmbeddingSize();

    int16_t* enroll_data = const_cast<int16_t*>(enroll_reader.data());
    const int enroll_samples = enroll_reader.num_sample();
    int16_t* test_data = const_cast<int16_t*>(test_reader.data());
    const int test_samples = test_reader.num_sample();

    std::vector<float> enroll_emb(embedding_size, 0.f);
    std::vector<float> test_emb(embedding_size, 0.f);
    speaker_engine->ExtractEmbedding(enroll_data, enroll_samples, &enroll_emb);
    speaker_engine->ExtractEmbedding(test_data, test_samples, &test_emb);

    const float score = speaker_engine->CosineSimilarity(enroll_emb, test_emb);
    const float same = (score >= threshold) ? 1.f : 0.f;
    return MakeFloatArray(env, score, same);
  } catch (const std::exception& e) {
    ThrowIo(env, e.what());
    return nullptr;
  } catch (...) {
    ThrowIo(env, "native 推理异常");
    return nullptr;
  }
}
