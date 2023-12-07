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

#ifndef SPEAKER_SPEAKER_MODEL_H_
#define SPEAKER_SPEAKER_MODEL_H_

#include <string>
#include <vector>

#include "utils/utils.h"

namespace wespeaker {

class SpeakerModel {
 public:
  virtual ~SpeakerModel() = default;
  // extract embedding
  // NOTE: https://www.cnblogs.com/zhmlzhml/p/12973618.html
  virtual void ExtractEmbedding(const std::vector<std::vector<float>>& feats,
                                std::vector<float>* embed) {}
};

}  // namespace wespeaker

#endif  // SPEAKER_SPEAKER_MODEL_H_
