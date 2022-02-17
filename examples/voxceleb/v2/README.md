## Results
* Setup: fbank80, num_frms200, aug_prob0.6, speed perturb, ArcMargin, SGD (no spec_aug)

| Model | Params | TEST O    | TEST E   | TEST H     |
|:------|--------|-----------|----------|------------|
| XVEC-TSTP-emb512 | 4.610M | 1.941%   | 1.896%    | 3.314%    |
| ResNet34-TSTP-emb256 | 6.700M | **1.000%**    | **1.067%**    | **1.990%**    |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192 | 6.190M | 1.090%    | 1.207%    | 2.280%    |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 20.761M |     |     |     |
