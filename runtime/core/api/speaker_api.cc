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
#include <iostream>
#include <utility>

#include "glog/logging.h"
#include "utils/utils.h"
#include "speaker/e2e_speaker.h"
#include "api/speaker_api.h"


#define PARAM_NULL_RETURN(param)            \
  if (param == nullptr) {                   \
    LOG(ERROR) << #param << " is nullptr";  \
    return ERRCODE_SPEAKER_INVALID_INPUT;   \
  }

void* wespeaker_init(const char* model_path,
                     const int feat_dim,
                     const int sample_rate,
                     const int embedding_size,
                     const int SamplePerChunk,
                     int* errcode) {
  if (model_path == nullptr) {
    LOG(ERROR) << "model_path is nullptr";
    (*errcode) = ERRCODE_SPEAKER_MODEL_FILE;
    return nullptr;
  }
  auto model = new wespeaker::E2ESPEAKER(
    model_path, feat_dim, sample_rate, embedding_size, SamplePerChunk);
  if (model == nullptr) {
    LOG(ERROR) << "creat new speaker error";
    (*errcode) = ERRCODE_SPEAKER_INIT_MODEL;
    return nullptr;
  }
  (*errcode) = ERRCODE_SPEAKER_SUCC;
  return reinterpret_cast<void*>(model);
}

void wespeaker_free(void* speaker) {
  auto model = reinterpret_cast<wespeaker::E2ESPEAKER*>(speaker);
  delete model;
}

int wespeaker_embedding_size(void* speaker) {
  auto model = reinterpret_cast<wespeaker::E2ESPEAKER*>(speaker);
  return model->EmbeddingSize();
}

int wespeaker_extract_embedding(void* speaker,
                                const char* data,
                                const int data_size,
                                float* embedding,
                                int embedding_size) {
  PARAM_NULL_RETURN(speaker);
  if (data == nullptr) {
    return ERRCODE_SPEAKER_INVALID_INPUT;
  }
  if (data_size / sizeof(int16_t) < 3200) {
    // If the speech duration is less than 0.2s,
    // it will not be processed and the error will be returned.
    return ERRCODE_SPEAKER_SHORT_INPUT;
  }
  auto model = reinterpret_cast<wespeaker::E2ESPEAKER*>(speaker);
  std::vector<float> embs;
  model->ExtractEmbedding(reinterpret_cast<const int16_t*>(data),
                          data_size / sizeof(int16_t),
                          &embs);
  if (embedding_size != embs.size()) {
    return ERRCODE_SPEAKER_INVALID_OUTPUT;
  }
  memcpy(embedding, embs.data(), model->EmbeddingSize() * sizeof(float));
  return ERRCODE_SPEAKER_SUCC;
}

float wespeaker_compute_similarity(float* embedding1,
                                   float* embedding2,
                                   void* speaker) {
  int embedding_size = wespeaker_embedding_size(speaker);
  std::vector<float> emb1(embedding1, embedding1 + embedding_size);
  std::vector<float> emb2(embedding2, embedding2 + embedding_size);
  auto model = reinterpret_cast<wespeaker::E2ESPEAKER*>(speaker);
  return model->CosineSimilarity(emb1, emb2);
}
