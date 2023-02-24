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


#ifndef API_SPEAKER_API_H_
#define API_SPEAKER_API_H_

#ifdef __cplusplus
extern "C" {
#endif

enum en_rec_err_code {
  ERRCODE_SPEAKER_SUCC = 0x10000,         // Success
  ERRCODE_SPEAKER_INVALID_INPUT = 0x10001,   // Invalid input parameter
  ERRCODE_SPEAKER_SHORT_INPUT = 0x10002,   // Speech duration is too short
  ERRCODE_SPEAKER_INIT_MODEL = 0x10003,  // Failed to initialize the model
  ERRCODE_SPEAKER_MODEL_FILE = 0x10004,  // Model file does not exist
  ERRCODE_SPEAKER_INVALID_OUTPUT = 0x10005,  // Invalid output
};

/// @brief Initialize model
/// @param[in] model_path  the path of model
/// @param[in] feat_dim  the dimension of speech feature
/// @param[in] sample_rate  sampling rate
/// @param[in] embedding_size  embedding size
/// @param[in] SamplePerChunk  samples of one chunk
/// @param[in] errcode   Error code
/// @return  speaker model
void* wespeaker_init(const char* model_path,
                     const int feat_dim,
                     const int sample_rate,
                     const int embedding_size,
                     const int SamplePerChunk,
                     int* errcode)
    __attribute__((visibility("default")));

/// @brief Release resources
/// @param speaker[in] speaker model
void wespeaker_free(void* speaker)
    __attribute__((visibility("default")));

/// @brief get the embedding size
/// @param speaker[in] speaker model
/// @return embedding size
int wespeaker_embedding_size(void* speaker)
    __attribute__((visibility("default")));

/// @brief extract embedding
/// @param[in] speaker speaker model
/// @param[in] data input data
/// @param[in] data_size the size of input data
/// @param[in] embedding the pointer to embedding
/// @param[in] embedding_size the size of embedding
int wespeaker_extract_embedding(void* speaker,
                                const char* data,
                                const int data_size,
                                float* embedding,
                                int embedding_size)
    __attribute__((visibility("default")));

/// @brief Calculate the cosine similarity between two embedding
/// @param embedding1[in]
/// @param embedding2[in]
/// @param speaker[in] speaker model
float wespeaker_compute_similarity(float* embedding1,
                                   float* embedding2,
                                   void* speaker)
    __attribute__((visibility("default")));

#ifdef __cplusplus
};
#endif

#endif  // API_SPEAKER_API_H_
