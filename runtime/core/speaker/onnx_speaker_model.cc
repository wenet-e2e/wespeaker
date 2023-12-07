// Copyright (c) 2023 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef USE_ONNX

#include <vector>

#include "glog/logging.h"
#include "speaker/onnx_speaker_model.h"
#include "utils/utils.h"

namespace wespeaker {

Ort::Env OnnxSpeakerModel::env_ =
    Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
Ort::SessionOptions OnnxSpeakerModel::session_options_ = Ort::SessionOptions();

void OnnxSpeakerModel::InitEngineThreads(int num_threads) {
  session_options_.SetIntraOpNumThreads(num_threads);
}

#ifdef USE_GPU
void OnnxSpeakerModel::SetGpuDeviceId(int gpu_id) {
  Ort::ThrowOnError(
      OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, gpu_id));
}
#endif

OnnxSpeakerModel::OnnxSpeakerModel(const std::string& model_path) {
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
// 1. Load sessions
#ifdef _MSC_VER
  speaker_session_ = std::make_shared<Ort::Session>(
      env_, ToWString(model_path).c_str(), session_options_);
#else
  speaker_session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                                    session_options_);
#endif
  // 2. Model info
  Ort::AllocatorWithDefaultOptions allocator;
  // 2.1. input info
  int num_nodes = speaker_session_->GetInputCount();
  // NOTE(cdliang): for speaker model, num_nodes is 1.
  CHECK_EQ(num_nodes, 1);
  input_names_.resize(num_nodes);
  char* name = speaker_session_->GetInputName(0, allocator);
  input_names_[0] = name;
  LOG(INFO) << "Ouput name: " << name;

  // 2.2. output info
  num_nodes = speaker_session_->GetOutputCount();
  CHECK_EQ(num_nodes, 1);
  output_names_.resize(num_nodes);
  name = speaker_session_->GetOutputName(0, allocator);
  output_names_[0] = name;
  LOG(INFO) << "Output name: " << name;
}

void OnnxSpeakerModel::ExtractEmbedding(
    const std::vector<std::vector<float>>& feats, std::vector<float>* embed) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // prepare onnx required data
  unsigned int num_frames = feats.size();
  unsigned int feat_dim = feats[0].size();
  std::vector<float> feats_onnx(num_frames * feat_dim, 0.0);
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t j = 0; j < feat_dim; ++j) {
      feats_onnx[i * feat_dim + j] = feats[i][j];
    }
  }
  // NOTE(cdliang): batchsize = 1
  const int64_t feats_shape[3] = {1, num_frames, feat_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, feats_onnx.data(), feats_onnx.size(), feats_shape, 3);
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(feats_ort));
  std::vector<Ort::Value> ort_outputs = speaker_session_->Run(
      Ort::RunOptions{nullptr}, input_names_.data(), inputs.data(),
      inputs.size(), output_names_.data(), output_names_.size());
  // output
  float* outputs = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();

  embed->reserve(type_info.GetElementCount());
  for (size_t i = 0; i < type_info.GetElementCount(); ++i) {
    embed->emplace_back(outputs[i]);
  }
}

}  // namespace wespeaker

#endif  // USE_ONNX
