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

#include <string>

#include "frontend/wav.h"
#include "gflags/gflags.h"
#include "speaker/speaker_engine.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_string(enroll_wav, "", "First wav as enroll wav.");
DEFINE_string(test_wav, "", "Second wav as test wav.");
DEFINE_double(threshold, 0.5, "Threshold");

DEFINE_string(speaker_model_path, "", "path of speaker model");
DEFINE_int32(fbank_dim, 80, "fbank feature dimension");
DEFINE_int32(sample_rate, 16000, "sample rate");
DEFINE_int32(embedding_size, 256, "embedding size");
DEFINE_int32(SamplesPerChunk, 32000, "samples of one chunk");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // init model
  LOG(INFO) << FLAGS_speaker_model_path;
  LOG(INFO) << "Init model ...";
  auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
      FLAGS_speaker_model_path, FLAGS_fbank_dim, FLAGS_sample_rate,
      FLAGS_embedding_size, FLAGS_SamplesPerChunk);
  int embedding_size = speaker_engine->EmbeddingSize();
  LOG(INFO) << "embedding size: " << embedding_size;
  // read enroll wav/pcm data
  auto data_reader = wenet::ReadAudioFile(FLAGS_enroll_wav);
  int16_t* enroll_data = const_cast<int16_t*>(data_reader->data());
  int enroll_samples = data_reader->num_sample();
  // NOTE(cdliang): memory allocation
  std::vector<float> enroll_embs(embedding_size, 0);
  int enroll_wave_dur = static_cast<int>(static_cast<float>(enroll_samples) /
                                         data_reader->sample_rate() * 1000);
  LOG(INFO) << enroll_wave_dur;
  speaker_engine->ExtractEmbedding(enroll_data, enroll_samples, &enroll_embs);
  // test wav
  auto test_data_reader = wenet::ReadAudioFile(FLAGS_test_wav);
  int16_t* test_data = const_cast<int16_t*>(test_data_reader->data());
  int test_samples = test_data_reader->num_sample();
  std::vector<float> test_embs(embedding_size, 0);
  int test_wave_dur = static_cast<int>(static_cast<float>(test_samples) /
                                       test_data_reader->sample_rate() * 1000);
  LOG(INFO) << test_wave_dur;
  speaker_engine->ExtractEmbedding(test_data, test_samples, &test_embs);
  float cosine_score;
  LOG(INFO) << "compute score ...";
  cosine_score = speaker_engine->CosineSimilarity(enroll_embs, test_embs);
  LOG(INFO) << "Cosine socre: " << cosine_score;
  if (cosine_score >= FLAGS_threshold) {
    LOG(INFO) << "It's the same speaker!";
  } else {
    LOG(INFO) << "Warning! It's a different speaker.";
  }
  return 0;
}
