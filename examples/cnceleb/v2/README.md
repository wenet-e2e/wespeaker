## Results

* Setup: fbank80, num_frms200, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* test_trials: CNC-Eval-Core.lst
* 🔥 UPDATE: We update this recipe according to the setups in the winning system of CNSRC 2022, and get obvious performance improvement compared with the old recipe. Check the [commit](https://github.com/wenet-e2e/wespeaker/pull/63/commits/b08804987b3bbb26f4963cedf634058474c743dd) for details.
    * LR scheduler warmup from 0
    * Remove one embedding layer

| Model                      | Params   | EER (%)   | minDCF (p=0.01) |
| :------------------------- | :------: | :-------: | :-------------: |
| OLD: ResNet34-TSTP-emb256-ep150 | 6.70M    | 8.426     | 0.487           |
| NEW: ResNet34-TSTP-emb256-ep165 | 6.63M    | 7.181     | 0.408           |
| NEW: ResNet34-TSTP-emb256-ep165+AS-Norm | 6.63M    | 6.899     | 0.367          |