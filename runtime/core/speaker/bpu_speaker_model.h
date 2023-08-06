// Copyright (c) 2023 Horizon Robotics (liangchengdong@mail.nwpu.edu.cn)
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

#ifndef SPEAKER_BPU_SPEAKER_MODEL_H_
#define SPEAKER_BPU_SPEAKER_MODEL_H_

#ifdef USE_BPU

#include <vector>
#include <string>
#include <memory>

#include "easy_dnn/data_structure.h"
#include "easy_dnn/model.h"

#include "speaker/speaker_model.h"

using hobot::easy_dnn::Model;
using hobot::easy_dnn::DNNTensor;

namespace wespeaker {

class BpuSpeakerModel : public SpeakerModel {
 public:
  BpuSpeakerModel() = default;
  explicit BpuSpeakerModel(const std::string& model_path);
  ~BpuSpeakerModel() = default;
  void ExtractEmbedding(const std::vector<std::vector<float>>& chunk_feats,
                        std::vector<float>* embed) override;
 private:
  void AllocMemory(
    std::vector<std::shared_ptr<DNNTensor>>* input_dnn_tensor_array,
    std::vector<std::shared_ptr<DNNTensor>>* output_dnn_tensor_array,
    Model* model);
  void Read(const std::string& model_path);
  void Reset();
  std::vector<std::shared_ptr<DNNTensor>> input_dnn_;
  std::vector<std::shared_ptr<DNNTensor>> output_dnn_;
  Model* speaker_dnn_handle_;
};

}  // namespace wespeaker

#endif  // USE_BPU
#endif  // SPEAKER_BPU_SPEAKER_MODEL_H_
