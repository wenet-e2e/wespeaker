// Copyright (c) 2023 Chengdong Liang (liangchengdongd@qq.com)
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

MnnSpeakerModel::MnnSpeakerModel(const std::string& model_path) {
  // 1. Load sessions
  speaker_interpreter_ = std::shared_ptr<MNN::Interpreter>(
      MNN::Interpreter::createFromFile(model_path.c_str()));
  speaker_interpreter_->setCacheFile(".cachefile");
  speaker_interpreter_->setSessionHint(MNN::Interpreter::MAX_TUNING_NUMBER, 5);

  MNN::ScheduleConfig config;
  config.type = MNN_FORWARD_CPU;
  config.numThread = thread_num_;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_Low;
  backend_config.power = MNN::BackendConfig::Power_High;
  config.backendConfig = &backend_config;

  speaker_session_ = speaker_interpreter_->createSession(config);
  if (!speaker_session_) {
    LOG(ERROR) << "Create session failed!";
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

  std::vector<int> input_dims = {1, static_cast<int>(num_frames),
                                 static_cast<int>(feat_dim)};

  // 1. input tensor
  auto inputTensor =
      speaker_interpreter_->getSessionInput(speaker_session_, nullptr);
  speaker_interpreter_->resizeTensor(inputTensor, input_dims);
  speaker_interpreter_->resizeSession(speaker_session_);

  std::shared_ptr<MNN::Tensor> nchwTensor(
      new MNN::Tensor(inputTensor, MNN::Tensor::CAFFE));
  for (size_t i = 0; i < num_frames; ++i) {
    for (size_t j = 0; j < feat_dim; ++j) {
      nchwTensor->host<float>()[i * feat_dim + j] = feats[i][j];
    }
  }
  // nchwTensor->host<float>() = feats.data();
  inputTensor->copyFromHostTensor(nchwTensor.get());

  // 2. run session
  speaker_interpreter_->runSession(speaker_session_);

  // 3. output
  auto output = speaker_interpreter_->getSessionOutput(speaker_session_, NULL);
  std::shared_ptr<MNN::Tensor> outputTensor(
      new MNN::Tensor(output, MNN::Tensor::CAFFE));
  output->copyToHostTensor(outputTensor.get());
  LOG(INFO) << "outputTensor->elementSize(): " << outputTensor->elementSize();
  embed->reserve(outputTensor->elementSize());
  for (int i = 0; i < outputTensor->elementSize(); ++i) {
    embed->push_back(outputTensor->host<float>()[i]);
  }
}

}  // namespace wespeaker

#endif  // USE_MNN
