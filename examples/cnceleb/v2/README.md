## Results

* Setup: fbank80, num_frms200, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* test_trials: CNC-Eval-Core.lst
* 🔥 UPDATE: We update this recipe according to the setups in the winning system of CNSRC 2022, and get obvious performance improvement compared with the old recipe. Check the [commit1](https://github.com/wenet-e2e/wespeaker/pull/63/commits/b08804987b3bbb26f4963cedf634058474c743dd), [commit2](https://github.com/wenet-e2e/wespeaker/pull/66/commits/6f6af29197f0aa0a5d1b1993b7feb2f41b97891f) for details.
    * LR scheduler warmup from 0
    * Remove one embedding layer
    * add large margin fine-tuning strategy (LM)

| Model                      | Params   | EER (%)   | minDCF (p=0.01) |
| :------------------------- | :------: | :-------: | :-------------: |
| OLD: ResNet34-TSTP-emb256-ep150 | 6.70M    | 8.426     | 0.487           |
| NEW: ResNet34-TSTP-emb256-ep150 | 6.63M    | 7.134     | 0.408           |
| NEW: ResNet34-TSTP-emb256-ep150+AS-Norm | 6.63M    | 6.747     | 0.367    |
| NEW: ResNet34-TSTP-emb256-ep150+LM | 6.63M    | 6.652     | 0.393         |
| NEW: ResNet34-TSTP-emb256-ep150+LM+AS-Norm | 6.63M    | **6.492**     | **0.354** |
| NEW: ECAPA_512-emb192-ep150 | 6.19M    | 8.313     | 0.432           |
| NEW: ECAPA_512-emb192-ep150+AS-Norm | 6.19M    | 7.644     | 0.390    |
| NEW: ECAPA_512-emb192-ep150+LM | 6.19M    | 8.004     | 0.422         |
| NEW: ECAPA_512-emb192-ep150+LM+AS-Norm | 6.19M    | 7.417     | 0.379 |
| NEW: ECAPA_1024-emb192-ep150 | 14.65M    | 7.879     | 0.420           |
| NEW: ECAPA_1024-emb192-ep150+AS-Norm | 14.65M    | 7.412     | 0.379    |
| NEW: ECAPA_1024-emb192-ep150+LM | 14.65M    | 7.986     | 0.417         |
| NEW: ECAPA_1024-emb192-ep150+LM+AS-Norm | 14.65M    | 7.395     | 0.372 |
