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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "frontend/wav.h"
#include "speaker/speaker_engine.h"
#include "utils/thread_pool.h"
#include "utils/timer.h"
#include "utils/utils.h"

DEFINE_string(wav_scp, "", "input wav scp");
DEFINE_string(wav_path, "", "input wav path");
DEFINE_string(result, "", "output embedding file");

DEFINE_string(speaker_model_path, "", "path of speaker model");
DEFINE_int32(fbank_dim, 80, "fbank feature dimension");
DEFINE_int32(sample_rate, 16000, "sample rate");
DEFINE_int32(embedding_size, 256, "embedding size");
DEFINE_int32(samples_per_chunk, 32000, "samples of one chunk");
DEFINE_int32(thread_num, 1, "num of extract_emb thread");

std::ofstream g_result;
std::mutex g_result_mutex;
int g_total_waves_dur = 0;
int g_total_extract_time = 0;

void extract_emb(std::pair<std::string, std::string> wav) {
  // init model
  auto speaker_engine = std::make_shared<wespeaker::SpeakerEngine>(
      FLAGS_speaker_model_path, FLAGS_fbank_dim, FLAGS_sample_rate,
      FLAGS_embedding_size, FLAGS_samples_per_chunk);
  int embedding_size = speaker_engine->EmbeddingSize();
  LOG(INFO) << "embedding size: " << embedding_size;
  // read wav.scp
  wenet::WavReader wav_reader(wav.second);
  CHECK_EQ(wav_reader.sample_rate(), 16000);
  int16_t* data = const_cast<int16_t*>(wav_reader.data());
  int samples = wav_reader.num_sample();
  // NOTE(cdliang): memory allocation
  std::vector<float> embs(FLAGS_embedding_size, 0);

  int wave_dur = static_cast<int>(static_cast<float>(samples) /
                                  wav_reader.sample_rate() * 1000);
  int extract_time = 0;
  wenet::Timer timer;
  speaker_engine->ExtractEmbedding(data, samples, &embs);
  extract_time = timer.Elapsed();
  LOG(INFO) << "process: " << wav.first
            << " RTF: " << static_cast<float>(extract_time) / wave_dur;
  g_result_mutex.lock();
  std::ostream& buffer = FLAGS_result.empty() ? std::cout : g_result;
  buffer << wav.first;
  for (size_t i = 0; i < embs.size(); i++) {
    buffer << " " << embs[i];
  }
  buffer << std::endl;
  g_total_waves_dur += wave_dur;
  g_total_extract_time += extract_time;
  g_result_mutex.unlock();
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_wav_scp.empty() && FLAGS_wav_path.empty()) {
    LOG(FATAL) << "wav_scp and wav_path should not be empty at the same time";
  }

  std::vector<std::pair<std::string, std::string>> waves;
  if (!FLAGS_wav_path.empty()) {
    waves.emplace_back(make_pair("test", FLAGS_wav_path));
  } else {
    std::ifstream wav_scp(FLAGS_wav_scp);
    std::string line;
    while (getline(wav_scp, line)) {
      std::vector<std::string> strs;
      wespeaker::SplitString(line, &strs);
      CHECK_EQ(strs.size(), 2);
      waves.emplace_back(make_pair(strs[0], strs[1]));
    }
    if (waves.empty()) {
      LOG(FATAL) << "Please provide non-empty wav scp.";
    }
  }

  if (!FLAGS_result.empty()) {
    g_result.open(FLAGS_result, std::ios::out);
  }

  {
    ThreadPool pool(std::min(FLAGS_thread_num, static_cast<int>(waves.size())));
    for (auto& wav : waves) {
      pool.enqueue(extract_emb, wav);
    }
  }
  LOG(INFO) << "Total: process " << g_total_waves_dur << "ms audio taken "
            << g_total_extract_time << "ms.";
  LOG(INFO) << "RTF: " << std::setprecision(4)
            << static_cast<float>(g_total_extract_time) / g_total_waves_dur;
  return 0;
}
