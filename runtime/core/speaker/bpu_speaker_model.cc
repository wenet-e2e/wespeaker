// Copyright (c) 2023 Horizon Robotics (liangchengdong@mail.nwpu.edu.cn)
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

#ifdef USE_BPU

#include "speaker/bpu_speaker_model.h"
#include <vector>
#include <cstring>
#include "glog/logging.h"

#include "easy_dnn/data_structure.h"
#include "easy_dnn/model_manager.h"
#include "easy_dnn/task_manager.h"

using hobot::easy_dnn::ModelManager;
using hobot::easy_dnn::Task;
using hobot::easy_dnn::TaskManager;


namespace wespeaker {

void BpuSpeakerModel::AllocMemory(
  std::vector<std::shared_ptr<DNNTensor>>* input_dnn_tensor_array,
  std::vector<std::shared_ptr<DNNTensor>>* output_dnn_tensor_array,
  Model* model) {
  int32_t input_counts = model->GetInputCount();
  LOG(INFO) << "input_counts: " << input_counts;
  input_dnn_tensor_array->resize(input_counts);
  // stage-1: input [1, 1, 198, 80]
  for (int32_t i = 0; i < input_counts; i++) {
    (*input_dnn_tensor_array)[i].reset(new DNNTensor);
    auto& input = (*input_dnn_tensor_array)[i];
    model->GetInputTensorProperties(input->properties, i);
    if (input->properties.tensorType != hbDNNDataType::HB_DNN_TENSOR_TYPE_F32) {
      LOG(FATAL) << "Input data type must be float32";
    }
    hbSysAllocCachedMem(&(input->sysMem[0]),
                          input->properties.alignedByteSize);
  }
  // stage-2: output
  int32_t output_counts = model->GetOutputCount();
  LOG(INFO) << "Output counts: " << output_counts;
  output_dnn_tensor_array->resize(output_counts);
  for (int32_t i = 0; i < output_counts; i++) {
    (*output_dnn_tensor_array)[i].reset(new DNNTensor);
    auto& output = (*output_dnn_tensor_array)[i];
    model->GetOutputTensorProperties(output->properties, i);
    if (output->properties.tensorType !=
        hbDNNDataType::HB_DNN_TENSOR_TYPE_F32) {
      LOG(FATAL) << "Output data type must be float32";
    }
    hbSysAllocCachedMem(&(output->sysMem[0]),
                        output->properties.alignedByteSize);
  }
}

void BpuSpeakerModel::Read(const std::string& model_path) {
  if (model_path == "") {
    LOG(FATAL) << "model_path muse set";
  }

  // Init bpu model
  ModelManager* model_manager = ModelManager::GetInstance();
  std::vector<Model*> models;
  int32_t ret_code = 0;
  // load speaker model
  // Model_path is bin model egs: speaker_resnet34.bin
  ret_code = model_manager->Load(models, model_path);
  if (ret_code != 0) {
      LOG(FATAL) << "easydn error code: "
                  << ", error loading bpu model speaker_model.bin at "
                  << model_path;
  }
  // get model handle
  speaker_dnn_handle_ = model_manager->GetModel([](Model* model) {
    LOG(INFO) << model->GetName();
    return model->GetName().find("speaker") != std::string::npos;
  });
  AllocMemory(&input_dnn_, &output_dnn_, speaker_dnn_handle_);
  Reset();
  LOG(INFO) << "Bpu Model Info:";
  LOG(INFO) << "Model_path:" << model_path;
}

BpuSpeakerModel::BpuSpeakerModel(const std::string& model_path) {
  this->Read(model_path);
}

void BpuSpeakerModel::ExtractEmbedding(
  const std::vector<std::vector<float>>& chunk_feats,
  std::vector<float>* embed) {
  // reset input && output
  Reset();
  // chunk_feats: [198, 80]
  auto& feat_input = input_dnn_[0];
  auto feat_ptr = reinterpret_cast<float*>(feat_input->sysMem[0].virAddr);
  int64_t addr_shift = 0;
  for (size_t i = 0; i < chunk_feats.size(); i++) {
    memcpy(feat_ptr + addr_shift, chunk_feats[i].data(),
           chunk_feats[i].size() * sizeof(float));
    addr_shift += chunk_feats[i].size();
  }

  hbSysFlushMem(&(feat_input->sysMem[0]), HB_SYS_MEM_CACHE_CLEAN);
  TaskManager* task_manager = TaskManager::GetInstance();
  auto infer_task = task_manager->GetModelInferTask(100);
  infer_task->SetModel(speaker_dnn_handle_);
  infer_task->SetInputTensors(input_dnn_);
  infer_task->SetOutputTensors(output_dnn_);
  infer_task->RunInfer();
  infer_task->WaitInferDone(100);
  infer_task.reset();

  hbSysFlushMem(&(output_dnn_[0]->sysMem[0]), HB_SYS_MEM_CACHE_INVALIDATE);
  int output_dim = \
    output_dnn_[0]->properties.validShape.dimensionSize[1];  // 256
  const float* raw_data = \
    reinterpret_cast<float*>(output_dnn_[0]->sysMem[0].virAddr);
  embed->reserve(output_dim);
  // NOTE(cdliang): default output_node = 1
  for (int idx = 0, i = 0; i < output_dim; i++) {
    embed->emplace_back(raw_data[idx++]);
  }
}

void BpuSpeakerModel::Reset() {
  auto set_to_zero =
    [](std::vector<std::shared_ptr<DNNTensor>>& input_dnn_tensor_array,
        std::vector<std::shared_ptr<DNNTensor>>& output_dnn_tensor_array) {
    for (auto& tensor : input_dnn_tensor_array) {
      memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
    }
    for (auto& tensor : output_dnn_tensor_array) {
      memset(tensor->sysMem[0].virAddr, 0, tensor->properties.alignedByteSize);
    }
  };
  set_to_zero(input_dnn_, output_dnn_);
}
}  // namespace wespeaker

#endif  // USE_BPU
