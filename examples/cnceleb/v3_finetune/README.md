## Fine-tuning Results Based on DINO

* Setup: fbank80, num_frms200, epoch75 (pretrain), epoch50 (finetune), ArcMargin, aug_prob0.6, speed_perturb (no spec_aug)
* [Pre-trained ECAPA-TDNN checkpoints](https://drive.google.com/drive/folders/1XDIUjnKPrvJE5auBWT5CcE4mqcglCwzq?usp=drive_link): teacher models extracted from `model_75.pt` (please refer to `wespeaker/ssl/bin/average_dino_model.py` for information on the extraction process)
* test_trials: CNC-Eval-Avg.lst
* These results are obtained by pretraining on different datasets and then finetuning with CNCeleb.


| Model                             | Params  |  FLOPs  |    Pretraining Data    | LM  | AS-Norm   | EER (%)   | minDCF (p=0.01)  |
| :------------------------------   | :-----: | :-----: | :--------------------: | :-: | :-------: | :-------: | :--------------: |
| ECAPA_TDNN_GLOB_c1024-ASTP-emb192 | 14.65M  | 2.65 G  |        CNCeleb         | ×   | ×         | 8.217     | 0.439            |
|                                   |         |         |                        | ×   | √         | 7.508     | 0.378            |
|                                   |         |         |                        | √   | ×         | 8.093     | 0.423            |
|                                   |         |         |                        | √   | √         | 7.339     | 0.366            |
|                                   |         |         | WenetSpeech (filtered) | ×   | ×         | 7.229     | 0.390            |
|                                   |         |         |                        | ×   | √         | 6.714     | 0.344            |
|                                   |         |         |                        | √   | ×         | 6.995     | 0.375            |
|                                   |         |         |                        | √   | √         | 6.474     | 0.331            |

* 🔥 UPDATE 2024.03: We support finetuning DINO-based self-supervised models, which is trained on the WenetSpeech dataset. Pretrained Paper related to the finetuning results:
    * [WenetSpeech: A 10000+ Hours Multi-domain Mandarin Corpus for Speech Recognition](https://arxiv.org/pdf/2110.03370.pdf)
    * [Leveraging In-the-wild Data for Effective Self-supervised Pretraining in Speaker Recognition](https://arxiv.org/pdf/2309.11730.pdf)
