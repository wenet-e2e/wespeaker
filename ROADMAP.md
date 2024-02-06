# Wespeaker Roadmap

## Version 2.0 (Time: 2023.12)

This is the roadmap for wespeaker version 2.0.

- [ ] SSL support
    - [ ] Algorithms
        - [x] DINO
        - [x] MOCO
        - [x] SimCLR
        - [ ] Iteratively psudo label prediction and supervised finetuning
    - [ ] Recipes
        - [x] VoxCeleb
        - [ ] WenetSpeech
        - [ ] Gigaspeech
- [ ] Recipes
    - [ ] 3D-speaker
    - [ ] NIST SRE
        - [x] SRE16
        - [ ] SRE18
  - [ ] Documents
    - [ ] Speaker embedding learning basics
    - [ ] Core code explanation
    - [ ] Step-by-step tutorials
      - [ ] VoxCeleb Supervised
      - [ ] VoxCeleb Self-supervised
      - [ ] VoxSRC Diarization

## Version 1.0 （Time: 2022.09)

This is the roadmap for wespeaker version 1.0.

- [x] Standard dataset support
    - [x] VoxCeleb
    - [x] CnCeleb
- [x] SOTA models support
    - [x] x-vector (tdnn based, milestone deep speaker embedding)
    - [x] r-vector (resnet based, winner of voxsrc 2019)
    - [x] ecapa-tdnn (variant of tdnn, winner of voxsrc 2020)
- [x] Back-end Support
    - [x] Cosine
    - [x] EER/minDCF
    - [x] AS-norm
    - [x] PLDA
- [x] UIO for effective industrial-scale dataset processing
    - [x] Online data augmentation
        - Noise && RIR
        - Speed Perturb
        - Specaug
- [x] ONNX support
- [x] Triton Server support (GPU)
- [ ] ~~
    - Training or finetuning big models such as WavLM might be too costly for
      current stage
- [x] Basic Speaker Diarization Recipe
    - Embedding based (more related with our speaker embedding learner toolkit)
- [x] Interactive Demo
    - Support using features from released pretrained models (hugging face)


## Current Support List
* Model (SOTA Models)
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/pdf/2005.07143.pdf)
    - [x] [RepVGG](https://arxiv.org/pdf/2101.03697.pdf)
    - [x] [CAM++](https://arxiv.org/pdf/2303.00332.pdf)
    - [x] [ERes2Net](https://arxiv.org/pdf/2305.12838.pdf)
* Pooling Functions
    - [x] TAP(mean) / TSDP(std) / TSTP(mean+std)
        - Comparison of mean/std pooling can be found in [shuai_iscslp](https://x-lance.sjtu.edu.cn/en/papers/2021/iscslp21_shuai_1_.pdf), [anna_arxiv](https://arxiv.org/pdf/2203.10300.pdf)
    - [x] Attentive Statistics Pooling (ASTP)
        - Mainly for ECAPA_TDNN
    - [x] Multi-Query and Multi-Head Attentive Statistics Pooling (MQMHASTP)
        - Details can be found in [MQMHASTP](https://arxiv.org/pdf/2110.05042.pdf)
* Criteria
    - [x] Softmax
    - [x] [Sphere (A-Softmax)](https://www.researchgate.net/publication/327389164)
    - [x] [Add_Margin (AM-Softmax)](https://arxiv.org/pdf/1801.05599.pdf)
    - [x] [Arc_Margin (AAM-Softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
    - [x] [Arc_Margin+Inter-topk+Sub-center](https://arxiv.org/pdf/2110.05042.pdf)
    - [x] [SphereFace2](https://ieeexplore.ieee.org/abstract/document/10094954)
* Scoring
    - [x] Cosine
    - [x] PLDA
    - [x] Score Normalization (AS-Norm)
* Metric
    - [x] EER
    - [x] minDCF
* Online Augmentation
    - [x] Noise && RIR
    - [x] Speed Perturb
    - [x] SpecAug
* Training Strategy
    - [x] Well-designed Learning Rate and Margin Schedulers
    - [x] Large Margin Fine-tuning
    - [x] Automatic Mixed Precision (AMP) Training
* Runtime
    - [x] Python Binding
    - [x] Triton Inference Server on verification && diarization in GPU deployment
    - [x] C++ Onnxruntime
* Self-Supervised Learning (SSL)
    - [x] [DINO](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
    - [x] [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)
    - [x] [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)
* Literature
    - [x] [Awesome Speaker Papers](docs/speaker_recognition_papers.md)
