# Copyright (c) 2022, Horizon Inc. Xingchen Song (sxc19@tsinghua.org.cn)
#               2023, Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import os
import sys
import random
import yaml

from torch.utils.data import DataLoader

from wespeaker.dataset.dataset import Dataset

try:
    import hbdk  # noqa: F401
    import horizon_nn  # noqa: F401
except ImportError:
    print('Please install hbdk,horizon_nn,horizon_tc_ui !')
    sys.exit(1)

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def make_calibration_data(args, conf, cal_data_dir):
    conf['shuffle'] = True
    logger.info(conf)
    dataset = Dataset(args.cali_data_type,
                      args.cali_datalist,
                      conf,
                      spk2id_dict={},
                      whole_utt=False,
                      reverb_lmdb_file=None,
                      noise_lmdb_file=None,
                      repeat_dataset=False)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=1,
                            num_workers=0)
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 100 == 0:
            logger.info("processed {} samples.".format(batch_idx))
        assert len(batch['key']) == 1
        key = batch['key'][0]  # [B, key]
        feat = batch['feat']
        feat = feat.unsqueeze(1).numpy()
        feats_save_path = os.path.join(cal_data_dir, '{}.bin'.format(key))
        feat.tofile(feats_save_path)


def generate_config(args):
    template = """
# 模型参数组
model_parameters:
  # 原始Onnx浮点模型文件
  onnx_model: '{}'
  # 转换的目标AI芯片架构
  march: 'bernoulli2'
  # 模型转换输出的用于上板执行的模型文件的名称前缀
  output_model_file_prefix: '{}'
  # 模型转换输出的结果的存放目录
  working_dir: '{}'
  # 指定转换后混合异构模型是否保留输出各层的中间结果的能力
  layer_out_dump: False
  # 转换过程中日志生成级别
  log_level: 'debug'
# 输入信息参数组
input_parameters:
  # 原始浮点模型的输入节点名称
  input_name: '{}'
  # 原始浮点模型的输入数据格式（数量/顺序与input_name一致）
  input_type_train: 'featuremap'
  # 原始浮点模型的输入数据排布（数量/顺序与input_name一致）
  input_layout_train: 'NCHW'
  # 原始浮点模型的输入数据尺寸
  input_shape: '{}'
  # 网络实际执行时，输入给网络的batch_size  默认值为1
  # input_batch: 1
  # 在模型中添加的输入数据预处理方法
  norm_type: 'no_preprocess'
  # 预处理方法的图像减去的均值; 如果是通道均值，value之间必须用空格分隔
  # mean_value: ''
  # 预处理方法的图像缩放比例，如果是通道缩放比例，value之间必须用空格分隔
  # scale_value: ''
  # 转换后混合异构模型需要适配的输入数据格式（数量/顺序与input_name一致）
  input_type_rt: 'featuremap'
  # 输入数据格式的特殊制式
  input_space_and_range: ''
  # 转换后混合异构模型需要适配的输入数据排布（数量/顺序与input_name一致）
  input_layout_rt: 'NCHW'
# 校准参数组
calibration_parameters:
  # 模型校准使用的标定样本的存放目录
  cal_data_dir: '{}'
  # 开启图片校准样本自动处理（skimage read resize到输入节点尺寸）
  preprocess_on: False
  # 校准使用的算法类型
  calibration_type: '{}'
  # max 校准方式的参数
  max_percentile: 1.0
  # 强制指定OP在CPU上运行
  run_on_cpu: '{}'
  # 强制指定OP在BPU上运行
  run_on_bpu: '{}'
# 编译参数组
compiler_parameters:
  # 编译策略选择
  compile_mode: 'latency'
  # 是否打开编译的debug信息
  debug: False
  # 模型运行核心数
  core_num: 1
  # 模型编译的优化等级选择
  optimize_level: 'O3'
"""
    output_dir = os.path.realpath(args.output_dir)
    speaker_onnx_path = os.path.realpath(args.onnx_path)
    speaker_log_path = os.path.join(output_dir, 'hb_makertbin_output_speaker')
    speaker_config = template.format(os.path.realpath(args.onnx_path),
                                     "speaker", speaker_log_path,
                                     args.input_name, args.input_shape,
                                     "cal_data_dir", args.calibration_type,
                                     args.extra_ops_run_on_cpu, "")
    with open(output_dir + "/config_speaker.yaml", "w") as speaker_yaml:
        speaker_yaml.write(speaker_config)


def get_args():
    parser = argparse.ArgumentParser(
        description='convert onnx to horizon .bin')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--cali_datalist',
                        type=str,
                        default=None,
                        help='make calibration data')
    parser.add_argument('--cali_data_type',
                        type=str,
                        default=None,
                        help='make calibration data')
    parser.add_argument('--extra_ops_run_on_cpu',
                        type=str,
                        default="",
                        help='extra operations running on cpu.')
    parser.add_argument('--calibration_type',
                        type=str,
                        default='default',
                        help='kl / max / default.')
    parser.add_argument('--onnx_path',
                        type=str,
                        required=True,
                        help='onnx model (float)')
    parser.add_argument('--input_name',
                        type=str,
                        required=True,
                        help='input name')
    parser.add_argument('--input_shape',
                        type=str,
                        required=True,
                        help='input shape')
    return parser


if __name__ == '__main__':
    random.seed(777)
    parser = get_args()
    args = parser.parse_args()
    os.system("mkdir -p " + args.output_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        conf = yaml.load(fin, Loader=yaml.FullLoader)

    cal_data_dir = os.path.join(args.output_dir, 'cal_data_dir')
    os.makedirs(cal_data_dir, exist_ok=True)

    logging.info("Stage-1: Generate config")
    generate_config(args)

    logging.info("Stage-2: Make calibration data")
    test_conf = copy.deepcopy(conf['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    elif 'mfcc_args' in test_conf:
        test_conf['mfcc_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['aug_prob'] = conf.get('aug_prob', 0.0)
    test_conf['filter'] = False
    make_calibration_data(args, test_conf, cal_data_dir)

    output_dir = os.path.realpath(args.output_dir)
    logger.info("Stage-3: Make speaker.bin")
    os.system("cd {} && mkdir -p hb_makertbin_log_speaker".format(output_dir) +
              " && cd hb_makertbin_log_speaker &&" +
              " hb_mapper makertbin --model-type \"onnx\" --config \"{}\"".
              format(output_dir + "/config_speaker.yaml"))
