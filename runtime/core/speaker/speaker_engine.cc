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

#include <algorithm>
#include <functional>
#include <limits>
#include "speaker/speaker_engine.h"


#ifdef USE_ONNX
  #include "speaker/onnx_speaker_model.h"
#endif

namespace wespeaker {

SpeakerEngine::SpeakerEngine(const std::string& model_path,
                             const int feat_dim,
                             const int sample_rate,
                             const int embedding_size,
                             const int SamplesPerChunk) {
  // NOTE(cdliang): default num_threads = 1
  const int kNumGemmThreads = 1;
  LOG(INFO) << "Reading model " << model_path;
  embedding_size_ = embedding_size;
  LOG(INFO) << "Embedding size: " << embedding_size_;
  per_chunk_samples_ = SamplesPerChunk;
  LOG(INFO) << "per_chunk_samples: " << per_chunk_samples_;
  feature_config_ = std::make_shared<wenet::FeaturePipelineConfig>(
    feat_dim, sample_rate);
  feature_pipeline_ = \
    std::make_shared<wenet::FeaturePipeline>(*feature_config_);
  feature_pipeline_->Reset();
#ifdef USE_ONNX
  OnnxSpeakerModel::InitEngineThreads(kNumGemmThreads);
  model_ = std::make_shared<OnnxSpeakerModel>(model_path);
#endif
}

int SpeakerEngine::EmbeddingSize() {
  return embedding_size_;
}

void SpeakerEngine::ApplyMean(std::vector<std::vector<float>>* feat,
                              unsigned int feat_dim) {
  std::vector<float> mean(feat_dim, 0);
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), mean.begin(),
                   std::plus<>{});
  }
  std::transform(mean.begin(), mean.end(), mean.begin(),
                 [&](const float d) {return d / feat->size();});
  for (auto& i : *feat) {
    std::transform(i.begin(), i.end(), mean.begin(), i.begin(), std::minus<>{});
  }
}

void SpeakerEngine::ExtractFeatureOneChunk(
  const std::vector<int16_t>& chunk_wav,
  std::vector<std::vector<float>>* chunk_feat) {
  feature_pipeline_->AcceptWaveform(chunk_wav);
  feature_pipeline_->set_input_finished();
  feature_pipeline_->Read(feature_pipeline_->num_frames(), chunk_feat);
  this->ApplyMean(chunk_feat, (*chunk_feat)[0].size());
  feature_pipeline_->Reset();
}

void SpeakerEngine::ExtractFeature(const int16_t* data, int data_size,
    std::vector<std::vector<std::vector<float>>>* chunks_feat) {
  if (data != nullptr) {
    std::vector<std::vector<float>> chunk_feat;
    if (per_chunk_samples_ <= 0 || per_chunk_samples_ == data_size) {
      // full mode
      this->ExtractFeatureOneChunk(std::vector<int16_t>(
        data, data + data_size), &chunk_feat);
      chunks_feat->push_back(chunk_feat);
      chunk_feat.clear();
    } else {
      // NOTE(cdliang): extract feature with chunk by chunk
      int chunk_num = static_cast<int>(data_size / per_chunk_samples_) + 1;
      for (int i = 1; i < chunk_num; i++) {
        this->ExtractFeatureOneChunk(std::vector<int16_t>(
          data + (i - 1) * per_chunk_samples_,
          data + i * per_chunk_samples_), &chunk_feat);
        chunks_feat->push_back(chunk_feat);
        chunk_feat.clear();
      }
      // last chunk
      std::vector<int16_t> chunk_wav(
        data + (chunk_num - 1) * per_chunk_samples_, data + data_size);
      chunk_wav.insert(chunk_wav.end(), data,
        data + per_chunk_samples_ - data_size % per_chunk_samples_);
      this->ExtractFeatureOneChunk(chunk_wav, &chunk_feat);
      chunks_feat->push_back(chunk_feat);
      chunk_feat.clear();
    }
  } else {
    LOG(ERROR) << "Input is nullptr!";
  }
}

void SpeakerEngine::ExtractEmbedding(const int16_t* data, int data_size,
                                     std::vector<float>* avg_emb) {
  // chunks_feat: [nchunk, T, D]
  std::vector<std::vector<std::vector<float>>> chunks_feat;
  this->ExtractFeature(data, data_size, &chunks_feat);
  int chunk_num = chunks_feat.size();
  avg_emb->resize(embedding_size_, 0);
  for (int i = 0; i < chunk_num; i++) {
    std::vector<float> tmp_emb;
    model_->ExtractEmbedding(chunks_feat[i], &tmp_emb);
    for (int j = 0; j < tmp_emb.size(); j++) {
      (*avg_emb)[j] += tmp_emb[j];
    }
  }
  // avg_emb: [embedding_size_]
  for (size_t i = 0; i < avg_emb->size(); i++) {
    (*avg_emb)[i] /= chunk_num;
  }
}

float SpeakerEngine::CosineSimilarity(const std::vector<float>& emb1,
                                      const std::vector<float>& emb2) {
  CHECK_EQ(emb1.size(), emb2.size());
  float dot = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0);
  float emb1_sum = std::inner_product(emb1.begin(), emb1.end(),
                                      emb1.begin(), 0.0);
  float emb2_sum = std::inner_product(emb2.begin(), emb2.end(),
                                      emb2.begin(), 0.0);
  dot /= std::max(std::sqrt(emb1_sum) * std::sqrt(emb2_sum),
                  std::numeric_limits<float>::epsilon());
  return dot;
}

}  // namespace wespeaker
