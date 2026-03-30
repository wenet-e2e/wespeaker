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

#include <sstream>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "speaker/onnx_speaker_model.h"
#include "utils/utils.h"

namespace wespeaker {

namespace {

int InferEmbeddingSizeFromOutputShape(const std::vector<int64_t>& shape) {
  if (shape.empty()) {
    return -1;
  }
  int64_t prod = 1;
  int num_positive = 0;
  for (int64_t d : shape) {
    if (d > 0) {
      prod *= d;
      ++num_positive;
    }
  }
  if (num_positive == static_cast<int>(shape.size())) {
    return static_cast<int>(prod);
  }
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (shape[i] > 0) {
      return static_cast<int>(shape[i]);
    }
  }
  return -1;
}

}  // namespace

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
  // 2. Model info：ORT 1.13+ 移除 GetInputName，仅保留 GetInputNameAllocated
  Ort::AllocatorWithDefaultOptions allocator;
  // 2.1. input info
  int num_nodes = static_cast<int>(speaker_session_->GetInputCount());
  // NOTE(cdliang): for speaker model, num_nodes is 1.
  CHECK_EQ(num_nodes, 1);
#if ORT_API_VERSION >= 13
  {
    auto name_ptr = speaker_session_->GetInputNameAllocated(0, allocator);
    input_name_strs_.emplace_back(name_ptr.get());
    input_names_.push_back(input_name_strs_.back().c_str());
  }
#else
  {
    char* name = speaker_session_->GetInputName(0, allocator);
    input_name_strs_.emplace_back(name != nullptr ? name : "");
    input_names_.push_back(input_name_strs_.back().c_str());
  }
#endif
  LOG(INFO) << "Input name: " << input_name_strs_[0];

  // 2.2. output info
  num_nodes = static_cast<int>(speaker_session_->GetOutputCount());
  CHECK_EQ(num_nodes, 1);
#if ORT_API_VERSION >= 13
  {
    auto name_ptr = speaker_session_->GetOutputNameAllocated(0, allocator);
    output_name_strs_.emplace_back(name_ptr.get());
    output_names_.push_back(output_name_strs_.back().c_str());
  }
#else
  {
    char* name = speaker_session_->GetOutputName(0, allocator);
    output_name_strs_.emplace_back(name != nullptr ? name : "");
    output_names_.push_back(output_name_strs_.back().c_str());
  }
#endif
  LOG(INFO) << "Output name: " << output_name_strs_[0];

  Ort::TypeInfo output_type = speaker_session_->GetOutputTypeInfo(0);
  // ORT 1.16+：Session 输出为 ConstTensorTypeAndShapeInfo，不可赋给
  // TensorTypeAndShapeInfo
  Ort::ConstTensorTypeAndShapeInfo tensor_shape =
      output_type.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> out_shape = tensor_shape.GetShape();
  std::ostringstream oss;
  for (size_t i = 0; i < out_shape.size(); ++i) {
    if (i) oss << ",";
    oss << out_shape[i];
  }
  LOG(INFO) << "ONNX output shape: [" << oss.str() << "]";

  int64_t elem_count = tensor_shape.GetElementCount();
  if (elem_count > 0) {
    embedding_size_ = static_cast<int>(elem_count);
  } else {
    embedding_size_ = InferEmbeddingSizeFromOutputShape(out_shape);
  }
  CHECK_GT(embedding_size_, 0)
      << "Cannot infer embedding size from ONNX output (shape may be fully "
         "dynamic); try exporting with static output shape or set embedding "
         "size explicitly.";
  LOG(INFO) << "Inferred embedding size from ONNX: " << embedding_size_;
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
