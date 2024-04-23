// Copyright (c) 2024 Chengdong Liang (liangchengdongd@qq.com)
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

#ifndef SPEAKER_MNN_SPEAKER_MODEL_H_
#define SPEAKER_MNN_SPEAKER_MODEL_H_

#ifdef USE_MNN

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "speaker/speaker_model.h"

namespace wespeaker {

class MnnSpeakerModel : public SpeakerModel {
 public:
  explicit MnnSpeakerModel(const std::string& model_path, int num_threads);

  void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                        std::vector<float>* embed) override;
  ~MnnSpeakerModel();

 private:
  // session
  std::shared_ptr<MNN::Interpreter> speaker_interpreter_;
  MNN::Session* speaker_session_ = nullptr;
};

}  // namespace wespeaker

#endif  // USE_MNN
#endif  // SPEAKER_MNN_SPEAKER_MODEL_H_
