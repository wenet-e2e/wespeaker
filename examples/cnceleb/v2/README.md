## Results

* Setup: fbank80, num_frms200, epoch150, ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* test_trials: CNC-Eval-Avg.lst
* 🔥 UPDATE 2022.07.12: We update this recipe according to the setups in the winning system of CNSRC 2022, and get obvious performance improvement compared with the old recipe. Check the [commit1](https://github.com/wenet-e2e/wespeaker/pull/63/commits/b08804987b3bbb26f4963cedf634058474c743dd), [commit2](https://github.com/wenet-e2e/wespeaker/pull/66/commits/6f6af29197f0aa0a5d1b1993b7feb2f41b97891f) for details.
    * LR scheduler warmup from 0
    * Remove one embedding layer
    * Add large margin fine-tuning strategy (LM)

| Model                             | Params    | FLOPs   | LM  | AS-Norm   | EER (%)   | minDCF (p=0.01)  |
| :------------------------------   | :-------: | :-----: | :-: | :-------: | :-------: | :--------------: |
| ResNet34-TSTP-emb256 (OLD)        | 6.70M     | 4.55 G  | ×   | ×         | 8.426     | 0.487            |
| ResNet34-TSTP-emb256              | 6.63M     | 4.55 G  | ×   | ×         | 7.134     | 0.408            |
|                                   |           |         | ×   | √         | 6.747     | 0.367            |
|                                   |           |         | √   | ×         | 6.652     | 0.393            |
|                                   |           |         | √   | √         | 6.492     | 0.354            |
| ResNet221-TSTP-emb256             | 23.86M    | 21.29 G | ×   | ×         | 5.965     | 0.362            |
|                                   |           |         | ×   | √         | 5.708     | **0.326**        |
|                                   |           |         | √   | ×         | 5.886     | 0.362            |
|                                   |           |         | √   | √         | **5.655** | 0.330            |
| ECAPA_TDNN_GLOB_c512-ASTP-emb192  | 6.19M     | 1.04 G  | ×   | ×         | 8.313     | 0.432            |
|                                   |           |         | ×   | √         | 7.644     | 0.390            |
|                                   |           |         | √   | ×         | 8.004     | 0.422            |
|                                   |           |         | √   | √         | 7.417     | 0.379            |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M    | 2.65 G  | ×   | ×         | 7.879     | 0.420            |
|                                   |           |         | ×   | √         | 7.412     | 0.379            |
|                                   |           |         | √   | ×         | 7.986     | 0.417            |
|                                   |           |         | √   | √         | 7.395     | 0.372            |
| RepVGG_TINY_A0                    | 6.26M     | 4.65 G  | ×   | ×         | 6.883     | 0.399            |
|                                   |           |         | ×   | √         | 6.550     | 0.355            |

