## Results for SRE16

* Setup: fbank40, num_frms200, epoch150, Softmax, aug_prob0.6
* Scoring: cosine & PLDA & PLDA Adaptation
* Metric: EER(%)

| Model                | Params | FLOPs  |  Backend   | Pooled | Tagalog | Cantonese |
|:---------------------|:------:|:------:|:----------:|:------:|:-------:|:---------:|
| ResNet34-TSTP-emb256 | 6.63M  | 4.55G  |   Cosine   |  15.4  |  19.82  |   10.39   |
|                      |        |        |    PLDA    | 11.689 | 16.961  |   6.239   |
|                      |        |        | Adapt PLDA | 5.788  |  8.974  |   2.674   |

Current PLDA implementation is fully compatible with the Kaldi version, note that
we can definitely improve the results with out adaptation with parameter tuning and extra LDA as shown in the Kaldi
Recipe, we didn't do this because we focus more on the adapted results, which are good enough under current setup.

* 🔥 UPDATE 2023.07.18: Support kaldi-compatible two-covariance PLDA and unsupervised domain adaptation.
* 🔥 UPDATE 2023.07.14: Support
  the [NIST SRE16 recipe](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016),
  see [#177](https://github.com/wenet-e2e/wespeaker/pull/177).
