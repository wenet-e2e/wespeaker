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

#ifndef SPEAKER_SPEAKER_ENGINE_H_
#define SPEAKER_SPEAKER_ENGINE_H_

#include <memory>
#include <string>
#include <vector>

#include "frontend/feature_pipeline.h"
#include "speaker/speaker_model.h"

namespace wespeaker {

class SpeakerEngine {
 public:
  explicit SpeakerEngine(const std::string& model_path, const int feat_dim,
                         const int sample_rate, const int embedding_size,
                         const int SamplesPerChunk);
  // return embedding_size
  int EmbeddingSize();
  // extract fbank
  void ExtractFeature(
      const int16_t* data, int data_size,
      std::vector<std::vector<std::vector<float>>>* chunks_feat);
  // extract embedding
  void ExtractEmbedding(const int16_t* data, int data_size,
                        std::vector<float>* avg_emb);

  float CosineSimilarity(const std::vector<float>& emb1,
                         const std::vector<float>& emb2);

 private:
  void ApplyMean(std::vector<std::vector<float>>* feats, unsigned int feat_dim);
  std::shared_ptr<wespeaker::SpeakerModel> model_ = nullptr;
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config_ = nullptr;
  std::shared_ptr<wenet::FeaturePipeline> feature_pipeline_ = nullptr;
  int embedding_size_ = 0;
  int per_chunk_samples_ = 32000;
  int sample_rate_ = 16000;
};

}  // namespace wespeaker

#endif  // SPEAKER_SPEAKER_ENGINE_H_
