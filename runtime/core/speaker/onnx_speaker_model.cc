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


#include <vector>

#include "speaker/onnx_speaker_model.h"
#include "glog/logging.h"


namespace wespeaker {

OnnxSpeakerModel::OnnxSpeakerModel(const std::string& model_path) {
  Ort::Env env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "OnnxModel");
  Ort::SessionOptions session_options_ = Ort::SessionOptions();
  session_options_.SetIntraOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  // 1. Load sessions
  speaker_session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                                      session_options_);
  // 2. Model info
  input_names_ = {"feats"};
  output_names_ = {"embs"};
}

void OnnxSpeakerModel::ExtractEmbedding(
  const std::vector<std::vector<float>>& feats,
  std::vector<float>* embed) {
  Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // prepare onnx required data
  std::vector<float> feats_onnx;
  unsigned int num_frames = feats.size();
  unsigned int feat_dim = feats[0].size();
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t j = 0; j < feat_dim; ++j) {
      feats_onnx.emplace_back(feats[i][j]);
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

  for (size_t i = 0; i < type_info.GetElementCount(); ++i) {
    embed->emplace_back(outputs[i]);
  }
}

}  // namespace wespeaker
