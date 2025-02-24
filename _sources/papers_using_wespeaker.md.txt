# Papers Implemented in WeSpeaker

[TOC]

## Stay Tuned! (Need to add a introduction for each paper)

## Introduction

After the release of the WeSpeaker project, many users from both academia and industry have actively engaged with it in their research. We appreciate all the feedback and contributions from the community and would like to highlight these interesting works.

Besides the citation of WeSpeaker itself, we highly recommend you to read and cite the corresponding papers as listed below.

```bibtex
@article{wang2024advancing,
  title={Advancing speaker embedding learning: Wespeaker toolkit for research and production},
  author={Wang, Shuai and Chen, Zhengyang and Han, Bing and Wang, Hongji and Liang, Chengdong and Zhang, Binbin and Xiang, Xu and Ding, Wen and Rohdin, Johan and Silnova, Anna and others},
  journal={Speech Communication},
  volume={162},
  pages={103104},
  year={2024},
  publisher={Elsevier}
}

@inproceedings{wang2023wespeaker,
  title={Wespeaker: A research and production oriented speaker embedding learning toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

## Architecture

### TDNN

```bibtex
@inproceedings{snyder2018x,
  title={X-vectors: Robust dnn embeddings for speaker recognition},
  author={Snyder, David and Garcia-Romero, Daniel and Sell, Gregory and Povey, Daniel and Khudanpur, Sanjeev},
  booktitle={2018 IEEE international conference on acoustics, speech and signal processing (ICASSP)},
  pages={5329--5333},
  year={2018},
  organization={IEEE}
}
```

### ECAPA-TDNN

```bibtex
@article{desplanques2020ecapa,
  title={Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification},
  author={Desplanques, Brecht and Thienpondt, Jenthe and Demuynck, Kris},
  journal={arXiv preprint arXiv:2005.07143},
  year={2020}
}
```

### Xi-vector

```bibtex
@article{lee2021xi,
  title={Xi-vector embedding for speaker recognition},
  author={Lee, Kong Aik and Wang, Qiongqiong and Koshinaka, Takafumi},
  journal={IEEE Signal Processing Letters},
  volume={28},
  pages={1385--1389},
  year={2021},
  publisher={IEEE}
}
```

### ResNet

The Current ResNet implementation is based on our system for VoxSRC2019, it's also the default speaker model in Pyannote.audio diarization pipeline (https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)

```bibtex
@article{zeinali2019but,
  title={But system description to voxceleb speaker recognition challenge 2019},
  author={Zeinali, Hossein and Wang, Shuai and Silnova, Anna and Mat{\v{e}}jka, Pavel and Plchot, Old{\v{r}}ich},
  journal={arXiv preprint arXiv:1910.12592},
  year={2019}
}
```

### ReDimNet

>

```bibtex
@article{yakovlev2024reshape,
  title={Reshape Dimensions Network for Speaker Recognition},
  author={Yakovlev, Ivan and Makarov, Rostislav and Balykin, Andrei and Malov, Pavel and Okhotnikov, Anton and Torgashov, Nikita},
  journal={arXiv preprint arXiv:2407.18223},
  year={2024}
}
```

### Golden gemini DF-ResNet

```bibtex
@article{liu2024golden,
  title={Golden gemini is all you need: Finding the sweet spots for speaker verification},
  author={Liu, Tianchi and Lee, Kong Aik and Wang, Qiongqiong and Li, Haizhou},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  year={2024},
  publisher={IEEE}
}
```

### SimAM-ResNet

```bibtex
@inproceedings{qin2022simple,
  title={Simple attention module based speaker verification with iterative noisy label detection},
  author={Qin, Xiaoyi and Li, Na and Weng, Chao and Su, Dan and Li, Ming},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={6722--6726},
  year={2022},
  organization={IEEE}
}
```

### Whisper based Speaker Verification

```bibtex
@article{zhao2024whisperpmfapartialmultiscalefeature,
      title={Whisper-PMFA: Partial Multi-Scale Feature Aggregation for Speaker Verification using Whisper Models},
      author={Yiyang Zhao and Shuai Wang and Guangzhi Sun and Zehua Chen and Chao Zhang and Mingxing Xu and Thomas Fang Zheng},
      year={2024},
      eprint={2408.15585},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2408.15585},
}
```

### CAM++

```bibtex
@article{wang2023cam++,
  title={Cam++: A fast and efficient network for speaker verification using context-aware masking},
  author={Wang, Hui and Zheng, Siqi and Chen, Yafeng and Cheng, Luyao and Chen, Qian},
  journal={arXiv preprint arXiv:2303.00332},
  year={2023}
}
```

### ERes2Net

```bibtex
@article{chen2023enhanced,
  title={An enhanced res2net with local and global feature fusion for speaker verification},
  author={Chen, Yafeng and Zheng, Siqi and Wang, Hui and Cheng, Luyao and Chen, Qian and Qi, Jiajun},
  journal={arXiv preprint arXiv:2305.12838},
  year={2023}
}
```

## Pipelines

### DINO Pretraining with Large-scale Data

```bibtex
@inproceedings{wang2024leveraging,
  title={Leveraging In-the-Wild Data for Effective Self-Supervised Pretraining in Speaker Recognition},
  author={Wang, Shuai and Bai, Qibing and Liu, Qi and Yu, Jianwei and Chen, Zhengyang and Han, Bing and Qian, Yanmin and Li, Haizhou},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10901--10905},
  year={2024},
  organization={IEEE}
}
```

## Dataset

### VoxBlink

```bibtex
@inproceedings{lin2024voxblink,
  title={Voxblink: A large scale speaker verification dataset on camera},
  author={Lin, Yuke and Qin, Xiaoyi and Zhao, Guoqing and Cheng, Ming and Jiang, Ning and Wu, Haiying and Li, Ming},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={10271--10275},
  year={2024},
  organization={IEEE}
}

@article{lin2024voxblink2,
  title={VoxBlink2: A 100K+ Speaker Recognition Corpus and the Open-Set Speaker-Identification Benchmark},
  author={Lin, Yuke and Cheng, Ming and Zhang, Fulin and Gao, Yingying and Zhang, Shilei and Li, Ming},
  journal={arXiv preprint arXiv:2407.11510},
  year={2024}
}
```

### VoxCeleb

```bibtex
@article{nagrani2017voxceleb,
  title={Voxceleb: a large-scale speaker identification dataset},
  author={Nagrani, Arsha and Chung, Joon Son and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1706.08612},
  year={2017}
}

@article{chung2018voxceleb2,
  title={Voxceleb2: Deep speaker recognition},
  author={Chung, Joon Son and Nagrani, Arsha and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1806.05622},
  year={2018}
}
```

### CNCeleb

```bibtex
@inproceedings{fan2020cn,
  title={Cn-celeb: a challenging chinese speaker recognition dataset},
  author={Fan, Yue and Kang, JW and Li, LT and Li, KC and Chen, HL and Cheng, ST and Zhang, PY and Zhou, ZY and Cai, YQ and Wang, Dong},
  booktitle={ICASSP 2020-2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7604--7608},
  year={2020},
  organization={IEEE}
}
```
