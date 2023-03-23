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
#include <fstream>
#include <sstream>
#include <iostream>

#include "frontend/wav.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include "speaker/speaker_engine.h"

DEFINE_string(wav_list, "", "input wav scp");
DEFINE_string(result, "", "output embedding file");

DEFINE_string(speaker_model_path, "", "path of speaker model");
DEFINE_int32(fbank_dim, 80, "fbank feature dimension");
DEFINE_int32(sample_rate, 16000, "sample rate");
DEFINE_int32(embedding_size, 256, "embedding size");
DEFINE_int32(SamplesPerChunk, 32000, "samples of one chunk");

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  // init model
  LOG(INFO) << "Init model ...";
  auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
    FLAGS_speaker_model_path, FLAGS_fbank_dim, FLAGS_sample_rate,
    FLAGS_embedding_size, FLAGS_SamplesPerChunk);
  int embedding_size = speaker_engine->EmbeddingSize();
  LOG(INFO) << "embedding size: " << embedding_size;
  // read wav.scp
  // [utt, wav_path]
  std::vector<std::pair<std::string, std::string>> waves;
  std::ifstream wav_scp(FLAGS_wav_list);
  std::string line;
  while (getline(wav_scp, line)) {
    std::vector<std::string> strs;
    wespeaker::SplitString(line, &strs);
    CHECK_EQ(strs.size(), 2);
    waves.emplace_back(make_pair(strs[0], strs[1]));
  }

  std::ofstream result;
  if (!FLAGS_result.empty()) {
    result.open(FLAGS_result, std::ios::out);
  }
  std::ostream &buffer = FLAGS_result.empty() ? std::cout : result;

  int total_waves_dur = 0;
  int total_extract_time = 0;
  for (auto &wav : waves) {
    auto data_reader = wenet::ReadAudioFile(wav.second);
    CHECK_EQ(data_reader->sample_rate(), 16000);
    int16_t* data = const_cast<int16_t*>(data_reader->data());
    int samples = data_reader->num_sample();
    // NOTE(cdliang): memory allocation
    std::vector<float> embs(embedding_size, 0);
    result << wav.first;

    int wave_dur = static_cast<int>(static_cast<float>(samples) /
                                    data_reader->sample_rate() * 1000);
    int extract_time = 0;
    wenet::Timer timer;
    speaker_engine->ExtractEmbedding(data, samples, &embs);
    extract_time = timer.Elapsed();
    for (size_t i = 0; i < embs.size(); i++) {
      result << " " << embs[i];
    }
    result << std::endl;
    LOG(INFO) << "process: " << wav.first
              << " RTF: " << static_cast<float>(extract_time) / wave_dur;
    total_waves_dur += wave_dur;
    total_extract_time += extract_time;
  }
  result.close();
  LOG(INFO) << "Total: process " << total_waves_dur << "ms audio taken "
            << total_extract_time << "ms.";
  LOG(INFO) << "RTF: "
            << static_cast<float>(total_extract_time) / total_waves_dur;
  return 0;
}
