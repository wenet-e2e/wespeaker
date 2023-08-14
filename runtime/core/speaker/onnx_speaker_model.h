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

#ifndef SPEAKER_ONNX_SPEAKER_MODEL_H_
#define SPEAKER_ONNX_SPEAKER_MODEL_H_

#ifdef USE_ONNX

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "speaker/speaker_model.h"

namespace wespeaker {

class OnnxSpeakerModel : public SpeakerModel {
 public:
  static void InitEngineThreads(int num_threads = 1);
#ifdef USE_GPU
  static void SetGpuDeviceId(int gpu_id = 0);
#endif
 public:
  explicit OnnxSpeakerModel(const std::string& model_path);

  void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                        std::vector<float>* embed) override;

 private:
  // session
  static Ort::Env env_;
  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> speaker_session_ = nullptr;
  // node names
  std::vector<const char*> input_names_;
  std::vector<const char*> output_names_;
  int embedding_size_ = 0;
};

}  // namespace wespeaker

#endif  // USE_ONNX
#endif  // SPEAKER_ONNX_SPEAKER_MODEL_H_
