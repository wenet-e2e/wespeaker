## SV Tutorial on VoxCeleb v3 (Self-Supervised)

If you meet any problems when going through this tutorial, please feel free to ask in
github [issues](https://github.com/wenet-e2e/wespeaker/issues). Thanks for any kind of feedback.

### First Experiment

We provide three self-supervised recipes on voxceleb data. They are currently the three most commonly used frameworks
for self-supervised speaker verification. If you want to learn more, you can refer to the `README.md` in the
corresponding directories.

* SimCLR: `examples/voxceleb/v3/simclr/run.sh`
* MoCo: `examples/voxceleb/v3/moco/run.sh`
* DINO: `examples/voxceleb/v3/dino/run.sh`

Because the steps of these three algorithms are basically the same, the following tutorial will take **DINO** as an
example. The recipe is simple and we suggest you run each stage one by one manually and check the result to understand
the whole processs.

```
cd examples/voxceleb/v3/dino
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5
bash run.sh --stage 6 --stop_stage 6
```

### Stage 1: Download Data

```
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi
```

This step is exactly the same as the recipe for supervised training on voxceleb `examples/voxceleb/v2`. **If you have
done it before, you can skip this step.**

This stage prepares the **voxceleb1**, **voxceleb2**, **MUSAN** and **RIRS_NOISES** dataset. MUSAN is a noise dataset
and RIRS_NOISES is a reverberation dataset, which are used for data augmentation. It should be noted that for
self-supervised speaker verification, data augmentation is crucial for the training process. We strongly recommend
incorporating MUSAN and RIRS_NOISES data augmentation here.

It should be noted that the `./local/prepare_data.sh` script starts from the stage 2. It is because the data downloading
process in stage 1 will take a long time. Thus we recommand you to download all archives above in your own way first and
put it under `data/download_data` and then run the above script.

When finishing this stage, you will get the following meta files:

* **wav.scp** files for all the dataset:
    * `data/musan/wav.scp`
    * `data/rirs/wav.scp`
    * `data/vox1/wav.scp`
    * `data/vox2_dev/wav.scp`
* **utt2spk** and **spk2utt** files for voxceleb1 and voxceleb2_dev
    * `data/vox1/utt2spk`
    * `data/vox1/spk2utt`
    * `data/vox2_dev/utt2spk`
    * `data/vox2_dev/spk2utt`
* **trials**
    * `data/vox1/trials/vox1_O_cleaned.kaldi`
    * `data/vox1/trials/vox1_E_cleaned.kaldi`
    * `data/vox1/trials/vox1_H_cleaned.kaldi`

**wav.scp** each line records two blank-separated columns : `wav_id` and `wav_path`

```
id10001/1zcIwhmdeo4/00001.wav /exported/data/voxceleb1_wav_v2/id10001/1zcIwhmdeo4/00001.wav
id10001/1zcIwhmdeo4/00002.wav /exported/data/voxceleb1_wav_v2/id10001/1zcIwhmdeo4/00002.wav
...
```

**utt2spk** each line records two blank-separated columns : `wav_id` and `spk_id`

```
id10001/1zcIwhmdeo4/00001.wav id10001
id10001/1zcIwhmdeo4/00002.wav id10001
...
```

**spk2utt** each line records many blank-separated columns : `spk_id` and many `wav_id`s belong to this `spk_id`

```
id10001 id10001/1zcIwhmdeo4/00001.wav id10001/1zcIwhmdeo4/00002.wav id10001/1zcIwhmdeo4/00003.wav ...
id10002 id10002/0_laIeN-Q44/00001.wav id10002/6WO410QOeuo/00001.wav ...
...
```

**trials** each line records three blank-separated columns : `enroll_wav_id`, `test_wav_id` and `label`

```
id10001/Y8hIVOBuels/00001.wav id10001/1zcIwhmdeo4/00001.wav target
id10001/Y8hIVOBuels/00001.wav id10943/vNCVj7yLWPU/00005.wav nontarget
id10001/Y8hIVOBuels/00001.wav id10001/7w0IBEWc9Qw/00004.wav target
id10001/Y8hIVOBuels/00001.wav id10999/G5R2-Hl7YX8/00008.wav nontarget
...
```

In this step, we generated **utt2spk** and **spk2utt**, but we will not use any speaker labels during the training
process.

### Stage 2: Reformat the Data

```
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi
```

This step is exactly the same as the recipe for supervised training on voxceleb `examples/voxceleb/v2`. **If you have
done it before, you can skip this step.**

The voxceleb dataset contains millions of wav files. Frequently opening the large scale small files will cause the IO
bottleneck. By default, the wav files from voxceleb dataset will be restored to some large binary shard files and the
shard files' paths ared store in `$data/$dset/shard.list` file. In this script, the wav file number in each shard file
is set to `1000`.

Besides, the MUSAN and RIR_NOISES dataset are stored in LMDB format for fastly random-access in the training process.

### Stage 3: Neural Network training

```
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/ssl/bin/train_dino.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --wav_scp ${data}/vox2_dev/wav.scp \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi
```

The NN model is trained in this step.

* Multi-GPU mode

Here, the `torchrun` command is used to start `num_gpus` processes for pytorch DDP training. Set the gpus ids
using `gpus` local variable. For example, `gpus="[0,1]"`, two gpus will be used and the used gpu idx is 0 and 1.

* Model Initialization

By default, the model is randomly initialized. You can also use some pre-trained model's weight to initialize the model
by specify the `model_init` param in the config file.

* Resume training

If your experiment is terminated after running several epochs for some reasons (e.g. the GPU is accidentally used by
other people and is out-of-memory ), you could continue the training from a checkpoint model. Just find out the finished
epoch in `exp/your_exp/`, set `checkpoint=exp/your_exp/$n.pt` and run the `run.sh --stage 3`. Then the training will
continue from the $n+1.pt

* Config

The config of neural network structure, optimization parameter, loss parameters, and dataset can be set in a YAML format
file.

Besides, under `conf/`, we have provide the configuration for different models, like ecapa `conf/ecapa.yaml` and
resnet `conf/resnet34.yaml`.

* Self-supervised Training Related

It's should be noted that for MoCo and SimCLR, the python scripts should be `wespeaker/ssl/bin/train_contrastive.py`.

The biggest difference compared to supervised training recipe `examples/voxceleb/v2` is the way data is organized.
Because self supervised training cannot use real labels, it can only construct sample pairs for contrastive learning
through assumptions. (a) The segments cropped from the same utterance belong to the same speaker (b) The segments
cropped from different utterances belong to different speakers.

For self-suerpervised training recipes, **dataloader** is defined in `wespeaker/ssl/dataset/dataset.py`. Next, I will
briefly introduce the process of dataset.

Firstly, we define different data reading methods based on different data storage formats. And perform global and local
shuffling.

```
dataset = DataList(lists, shuffle=shuffle)
if data_type == 'shard':
    dataset = Processor(dataset, processor.url_opener)
    dataset = Processor(dataset, processor.tar_file_and_group)
elif data_type == 'raw':
    dataset = Processor(dataset, processor.parse_raw)
else:
    dataset = Processor(dataset, processor.parse_feat)
# Local shuffle
if shuffle:
    dataset = Processor(dataset, processor.shuffle,
                        **configs['shuffle_args'])
```

Then, we defined different sample pair composition methods for different training methods. For SimCLR and MoCo, we take
2 segments from each sentence randomly. For DINO, we will crop 2 short and 4 long segments to form a positive pair.

```
# random chunk
frame_shift = configs['fbank_args'].get('frame_shift', 10)
frame_length = configs['fbank_args'].get('frame_length', 25)
chunk_info_args = configs['chunk_info_args']
for key in chunk_info_args:
    if 'chunk_len' in key:
        chunk_info_args[key] = (
            (chunk_info_args[key] - 1) * frame_shift +
            frame_length) * resample_rate // 1000
chunk_info_args['data_type'] = data_type
dataset = Processor(dataset, ssl_processor.random_chunk_for_dino,
                    **chunk_info_args)
```

Finally, it is a very important data augmentation step. We will randomly apply different data augmentation strategies to
each segment here.

```
# add reverb & noise
aug_prob = configs.get('aug_prob', 0.6)
if (reverb_lmdb_file and noise_lmdb_file) and (aug_prob > 0.0):
    reverb_data = LmdbData(reverb_lmdb_file)
    noise_data = LmdbData(noise_lmdb_file)
    dataset = Processor(dataset, ssl_processor.add_reverb_noise,
                        reverb_data, noise_data, resample_rate,
                        aug_prob)
```

Wespeaker notably facilitates effortless configuration for organizing diverse processors into a pipeline, ensuring both
efficiency and ease of extension. And the SSL related processors are defined in `wespeaker/ssl/dataset/processor.py`

In addition, in order to be more compatible with the existing framework of WeSpeaker, we have added wrappers to the
training models of SimCLR, MoCo, and DINO, which are defined in `wespeaker/ssl/models`. It includes **additional modules
** required for SSL training, the definition of **loss functions** and so on.

### Stage 4: Speaker Embedding Extraction

```
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/ssl/bin/average_dino_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $avg_model \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi
```

The embeddings for the train and evaluation set are extracted in this stage.

* Average Model

Average the model's weights from last `num_avg` checkpoints. This is a kind of model ensamble strategy to improve the
system performance.

It's should be noted that for MoCo and SimCLR, the python scripts should
be `wespeaker/ssl/bin/average_contrastive_model.py`. Because self-supervised training frameworks generally require the
introduction of additional modules (such as student model, projection head et al.) to assist in training, it is
necessary to remove these additional modules in this step to facilitate subsequent feature extraction.

* Extract Embedding

The extracted embeddings are stored in `exp/your_exp/embeddings` in kaldi scp,ark format. If there is something wrong
happened in this stage, you can check the log files under `exp/your_exp/embeddings/log` directory.

### Stage 5: Scoring the Evaluation Set

```
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi
```

All the trails listed in local variable `trials` is scored in this stage. **Cosine similarity** is used to calculate the
score for each trial pair. At the end of this stage, the Equal Error rate (EER), minDCF evaluation results are stored in
the `exp/your_exp/scores/vox1_cos_result` file. Besides, the detailed score for each trial with trial_name `trial_xx`
can be found in `exp/your_exp/scores/trial_xx.score` file.

Unlike supervised training recipe, we will not perform asnorm here because theoretically we cannot use any voxceleb
labels for score normalization.

### Stage 6(Optional): Export the trained model

```
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi
```

`wenet/bin/export_jit.py` will export the trained model using Libtorch. The exported model files can be easily used for
C++ inference in our runtime.
