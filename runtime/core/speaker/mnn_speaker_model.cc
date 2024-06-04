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

#ifdef USE_MNN

#include <vector>

#include "glog/logging.h"
#include "speaker/mnn_speaker_model.h"
#include "utils/utils.h"

namespace wespeaker {

MnnSpeakerModel::MnnSpeakerModel(const std::string& model_path,
                                 int num_threads) {
  // 1. Load sessions
  speaker_interpreter_ = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path.c_str()));

  MNN::ScheduleConfig config;
  config.type = MNN_FORWARD_CPU;
  config.numThread = num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_Low;
  backend_config.power = MNN::BackendConfig::Power_High;
  config.backendConfig = &backend_config;

  speaker_session_ = speaker_interpreter_->createSession(config);
  if (!speaker_session_) {
    LOG(ERROR) << "[MNN] Create session failed!";
    return;
  }
}

MnnSpeakerModel::~MnnSpeakerModel() {
  if (speaker_session_) {
    speaker_interpreter_->releaseModel();
    speaker_interpreter_->releaseSession(speaker_session_);
  }
}

void MnnSpeakerModel::ExtractEmbedding(
    const std::vector<std::vector<float>>& feats, std::vector<float>* embed) {
  unsigned int num_frames = feats.size();
  unsigned int feat_dim = feats[0].size();

  // 1. input tensor
  auto input_tensor =
      speaker_interpreter_->getSessionInput(speaker_session_, nullptr);

  auto shape = input_tensor->shape();
  CHECK_EQ(shape.size(), 3);
  if (shape[0] == -1 || shape[1] == -1 || shape[2] == -1) {
    VLOG(2) << "dynamic shape.";
    std::vector<int> input_dims = {1, static_cast<int>(num_frames),
                                   static_cast<int>(feat_dim)};
    speaker_interpreter_->resizeTensor(input_tensor, input_dims);
    speaker_interpreter_->resizeSession(speaker_session_);
  } else {
    if (shape[0] != 1 || shape[1] != num_frames || shape[2] != feat_dim) {
      LOG(ERROR) << "shape error!";
      return;
    }
  }

  std::shared_ptr<MNN::Tensor> nchw_tensor(
      new MNN::Tensor(input_tensor, MNN::Tensor::CAFFE));  // NCHW
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t j = 0; j < feat_dim; ++j) {
      nchw_tensor->host<float>()[i * feat_dim + j] = feats[i][j];
    }
  }
  input_tensor->copyFromHostTensor(nchw_tensor.get());

  // 2. run session
  speaker_interpreter_->runSession(speaker_session_);

  // 3. output
  auto output = speaker_interpreter_->getSessionOutput(speaker_session_, NULL);
  std::shared_ptr<MNN::Tensor> output_tensor(
      new MNN::Tensor(output, MNN::Tensor::CAFFE));
  output->copyToHostTensor(output_tensor.get());
  embed->reserve(output_tensor->elementSize());
  for (int i = 0; i < output_tensor->elementSize(); ++i) {
    embed->push_back(output->host<float>()[i]);
  }
}

}  // namespace wespeaker

#endif  // USE_MNN
