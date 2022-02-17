## Results
* Setup: fbank80, aug_prob0.6, ArcMargin, SGD

| Model | rir+noise | speed perturb | spec aug  | TEST O    | TEST E   | TEST H     |
|-------|-----------|---------------|-----------|-----------|----------|------------|
| ECAPA_TDNN_emb192_channels512 | Yes   | No    | No    | 1.170%    | 1.221%    | 2.234%    |
| ECAPA_TDNN_emb192_channels512 | Yes   | Yes   | No    | 1.085%    | 1.205%    | 2.288%    |
| ResNet34_emb256 | Yes   | No    | No    | 1.000%    | 1.149%    | 2.094%    |
| ResNet34_emb256 | Yes   | Yes   | No    | **1.000%**    | **1.067%**    | **1.990%**    |
