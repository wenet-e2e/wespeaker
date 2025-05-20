## SV Tutorial on VoxCeleb v2 (Supervised)

If you meet any problems when going through this tutorial, please feel free to ask in
github [issues](https://github.com/wenet-e2e/wespeaker/issues). Thanks for any kind of feedback.

### First Experiment

We provide a recipe `examples/voxceleb/v2/run.sh` on voxceleb data.

The recipe is simple and we suggest you run each stage one by one manually and check the result to understand the whole
process.

```
cd examples/voxceleb/v2/
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5
bash run.sh --stage 6 --stop_stage 6
bash run.sh --stage 7 --stop_stage 7
bash run.sh --stage 8 --stop_stage 8
```

### Stage 1: Download Data

```
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi
```

This stage prepares the **voxceleb1**, **voxceleb2**, **MUSAN** and **RIRS_NOISES** dataset. MUSAN is a noise dataset
and RIRS_NOISES is a reverberation dataset, which are used for data augmentation.

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
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
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

Besides, under `conf/`, we have provide the configuration for different models, like ecapa, resnet, et al.

### Stage 4: Speaker Embedding Extraction

```
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model
  if [[ $config == *repvgg*.yaml ]]; then
    echo "convert repvgg model ..."
    python wespeaker/models/convert_repvgg.py \
      --config $exp_dir/config.yaml \
      --load $avg_model \
      --save $exp_dir/models/convert_model.pt
    model_path=$exp_dir/models/convert_model.pt
  fi

  echo "Extract embeddings ..."
  local/extract_vox.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi
```

The embeddings for the train and evaluation set are extracted in this stage.

* Average Model

Average the model's weights from last `num_avg` checkpoints. This is a kind of model ensamble strategy to improve the
system performance.

* RepVGG Model Convert

Because the RepVGG model have different forward paradigms for training and evaluation, here the model weight is
converted to evaluation format.

* Extract Embedding

The extracted embeddings are stored in `exp/your_exp/embeddings` in kaldi scp,ark format. If there is someting wrong
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

### Stage 6: Scoring the Evaluation Set

```
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi
```

Based on the score results in the last stage,
the [adaptaive score normalization](https://www.isca-speech.org/archive/pdfs/interspeech_2017/matejka17_interspeech.pdf)
is done to further improve the results. The final evaluation results are stored
in `exp/your_exp/scores/vox1_${}${top_n}_result` file.

`--score_norm_method`: asnorm or snorm, detailed algorithm can be found in
this [paper](https://www.isca-speech.org/archive/pdfs/interspeech_2017/matejka17_interspeech.pdf).
`--top_n`: the negative cohort size to calculate the adaptive statistics

### Stage 7(Optional): Export the trained model

```
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  # Export the best model you want
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
```

`wenet/bin/export_jit.py` will export the trained model using Libtorch. The exported model files can be easily used for
C++ inference in our runtime.

### Stage 8(Optional): Large Margin Finetuning

```
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
```

This [paper](https://arxiv.org/pdf/2010.11255.pdf) has shown that finetuning the model for another few epoches
by increasing the training segment duration and enlarging the margin in the loss function at the same time can further
improve the performance for Voxceleb data. Users can run this stage for the better results.
