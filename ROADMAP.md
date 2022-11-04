# Wespeaker Roadmap
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
        -  Noise && RIR
        -  Speed Perturb
        -  Specaug
- [x] ONNX support
- [x] Triton Server support (GPU)
- [ ] ~~
    - Training or finetuning big models such as WavLM might be too costly for current stage
- [x] Basic Speaker Diarization Recipe
    - Embedding based (more related with our speaker embedding learner toolkit)
- [x] Interactive Demo
    - Support using features from released pretrained models (hugging face)
