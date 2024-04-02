## Diarization Tutorial on VoxConverse v2

If you meet any problems when going through this tutorial, please feel free to ask in github [issues](https://github.com/wenet-e2e/wespeaker/issues). Thanks for any kind of feedback.


### First Experiment

Speaker diarization is a typical downstream task of applying the well-learnt speaker embedding.
Here we introduce our diarization recipe `examples/voxconverse/v2/run.sh` on the Voxconverse 2020 dataset.

Note that we provide two recipes: **v1** and **v2**. Their only difference is that in **v2**, we split the Fbank extraction, embedding extraction and clustering modules to different stages.
We recommend newcomers to follow the **v2** recipe and run it stage by stage and check the result to better understand the whole process.

```
cd examples/voxconverse/v2/
bash run.sh --stage 1 --stop_stage 1
bash run.sh --stage 2 --stop_stage 2
bash run.sh --stage 3 --stop_stage 3
bash run.sh --stage 4 --stop_stage 4
bash run.sh --stage 5 --stop_stage 5
bash run.sh --stage 6 --stop_stage 6
bash run.sh --stage 7 --stop_stage 7
bash run.sh --stage 8 --stop_stage 8
```


### Stage 1: Download Prerequisites

```
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools

    # [1] Download evaluation toolkit
    wget -c https://github.com/usnistgov/SCTK/archive/refs/tags/v2.4.12.zip -O external_tools/SCTK-v2.4.12.zip
    unzip -o external_tools/SCTK-v2.4.12.zip -d external_tools

    # [2] Download voice activity detection model pretrained by Silero Team
    wget -c https://github.com/snakers4/silero-vad/archive/refs/tags/v3.1.zip -O external_tools/silero-vad-v3.1.zip
    unzip -o external_tools/silero-vad-v3.1.zip -d external_tools

    # [3] Download ResNet34 speaker model pretrained by WeSpeaker Team
    mkdir -p pretrained_models

    wget -c https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx -O pretrained_models/voxceleb_resnet34_LM.onnx
fi
```

Download three Prerequisites:
* the evaluation toolkit **SCTK**: Compute the DER metric
* the open-source VAD model pre-trained by [silero-vad](https://github.com/snakers4/silero-vad): Remove the silence in audio
* the pre-trained ResNet34 model: Extract the speaker embeddings

When finishing this stage, you will get two new dirs:
- **external_tools**
    - SCTK-v2.4.12.zip
    - SCTK-v2.4.12
    - silero-vad-v3.1.zip
    - silero-vad-v3.1
- **pretrained_models**
    - voxceleb_resnet34_LM.onnx


### Stage 2: Download and Prepare Data

```
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p data

    # Download annotations for dev and test sets (version 0.0.3)
    wget -c https://github.com/joonson/voxconverse/archive/refs/heads/master.zip -O data/voxconverse_master.zip
    unzip -o data/voxconverse_master.zip -d data

    # Download annotations from VoxSRC-23 validation toolkit (looks like version 0.0.2)
    # cd data && git clone https://github.com/JaesungHuh/VoxSRC2023.git --recursive && cd -

    # Download dev audios
    mkdir -p data/dev

    wget --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip -O data/voxconverse_dev_wav.zip
    unzip -o data/voxconverse_dev_wav.zip -d data/dev

    # Create wav.scp for dev audios
    ls `pwd`/data/dev/audio/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/dev/wav.scp

    # Test audios
    mkdir -p data/test

    wget  --no-check-certificate -c https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_test_wav.zip -O data/voxconverse_test_wav.zip
    unzip -o data/voxconverse_test_wav.zip -d data/test

    # Create wav.scp for test audios
    ls `pwd`/data/test/voxconverse_test_wav/*.wav | awk -F/ '{print substr($NF, 1, length($NF)-4), $0}' > data/test/wav.scp
fi
```

Download the Voxconverse 2020 dev and test sets as well as their annotations.
Here we use the latest version 0.0.3 in default (recommended).
You can also try the version 0.0.2 (seem to be used in the [VoxSRC-23 baseline repo](https://github.com/JaesungHuh/VoxSRC2023.git)).

When finishing this stage, you will get the new **data** dir:
- **data**
    - voxconverse_master.zip
    - voxconverse_dev_wav.zip
    - voxconverse_test_wav.zip
    - voxconverse_master
        - dev: ground-truth rttms
        - test: ground-truth rttms
    - dev
        - audio: wav files
        - wav.scp
    - test
        - voxconverse_test_wav: wav files
        - wav.scp

**wav.scp**: each line records two blank-separated columns : `wav_id` and `wav_path`
```
abjxc /path/to/wespeaker/examples/voxconverse/v2/data/dev/audio/abjxc.wav
afjiv /path/to/wespeaker/examples/voxconverse/v2/data/dev/audio/afjiv.wav
...
```


### Stage 3: Apply SAD (i.e., VAD)

```
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # Set VAD min duration
    min_duration=0.255

    if [[ "x${sad_type}" == "xoracle" ]]; then
        # Oracle SAD: handling overlapping or too short regions in ground truth RTTM
        while read -r utt wav_path; do
            python3 wespeaker/diar/make_oracle_sad.py \
                    --rttm data/voxconverse-master/${partition}/${utt}.rttm \
                    --min-duration $min_duration
        done < data/${partition}/wav.scp > data/${partition}/oracle_sad
    fi

    if [[ "x${sad_type}" == "xsystem" ]]; then
       # System SAD: applying 'silero' VAD
       python3 wespeaker/diar/make_system_sad.py \
               --repo-path external_tools/silero-vad-3.1 \
               --scp data/${partition}/wav.scp \
               --min-duration $min_duration > data/${partition}/system_sad
    fi
fi
```

`sad_type` could be oracle or system:
* oracle: get vad infos from the ground truth RTTMs, saved in `data/${partition}/oracle_sad`
* system: compute vad results using the [silero-vad](https://github.com/snakers4/silero-vad), saved in `data/${partition}/system_sad`

where `partition` is dev or test.

Note that too short VAD segments with less than `min_duration` seconds are ignored and simply regarded as silence.


### Stage 4: Extract Fbank Features

```
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then

    [ -d "exp/${sad_type}_sad_fbank" ] && rm -r exp/${sad_type}_sad_fbank

    echo "Make Fbank features and store it under exp/${sad_type}_sad_fbank"
    echo "..."
    bash local/make_fbank.sh \
            --scp data/${partition}/wav.scp \
            --segments data/${partition}/${sad_type}_sad \
            --store_dir exp/${partition}_${sad_type}_sad_fbank \
            --subseg_cmn ${subseg_cmn} \
            --nj 24
fi
```

`subseg_cmn` suggests applying Cepstral Mean Normalization (CMN) to Fbanks:
* on the sliding-window sub-segment (`subseg_cmn=true`) or
* on the whole vad segment (`subseg_cmn=false`)

You can specify `nj` jobs according to your cpu cores num.
The final Fbank features are saved under dir `exp/${partition}_${sad_type}_sad_fbank`.


### Stage 5: Extract Sliding-window Speaker Embeddings

```
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then

    [ -d "exp/${sad_type}_sad_embedding" ] && rm -r exp/${sad_type}_sad_embedding

    echo "Extract embeddings and store it under exp/${sad_type}_sad_embedding"
    echo "..."
    bash local/extract_emb.sh \
            --scp exp/${partition}_${sad_type}_sad_fbank/fbank.scp \
            --pretrained_model pretrained_models/voxceleb_resnet34_LM.onnx \
            --device cuda \
            --store_dir exp/${partition}_${sad_type}_sad_embedding \
            --batch_size 96 \
            --frame_shift 10 \
            --window_secs 1.5 \
            --period_secs 0.75 \
            --subseg_cmn ${subseg_cmn} \
            --nj 1
fi
```

Extract speaker embeddings from the Fbank features in a sliding-window fashion: `step=0.75s, window=1.5s`,
which means extracting embedding from each `1.5s` speech window after every `0.75s`.
Thus the contiguous windows overlap by `1.5-0.75=0.75s` in duration.

You can also specify `nj` jobs and decide to use the `gpu` or `cpu` devices.
The extracted embeddings are saved under dir `exp/${partition}_${sad_type}_sad_embedding`.


### Stage 6: Apply Spectral Clustering

```
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    [ -f "exp/spectral_cluster/${partition}_${sad_type}_sad_labels" ] && rm exp/spectral_cluster/${partition}_${sad_type}_sad_labels

    echo "Doing spectral clustering and store the result in exp/spectral_cluster/${partition}_${sad_type}_sad_labels"
    echo "..."
    python3 wespeaker/diar/spectral_clusterer.py \
            --scp exp/${partition}_${sad_type}_sad_embedding/emb.scp \
            --output exp/spectral_cluster/${partition}_${sad_type}_sad_labels
fi
```

Apply spectral clustering using the extracted sliding-window speaker embeddings,
and store the results in `exp/spectral_cluster/${partition}_${sad_type}_sad_labels`,
where each line records two blank-separated columns : `subseg_id` and `spk_id`
```
abjxc-00000400-00007040-00000000-00000150 0
abjxc-00000400-00007040-00000075-00000225 0
abjxc-00000400-00007040-00000150-00000300 0
abjxc-00000400-00007040-00000225-00000375 0
...
```


### Stage 7: Reformat Clustering Labels into RTTMs

```
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
    python3 wespeaker/diar/make_rttm.py \
            --labels exp/spectral_cluster/${partition}_${sad_type}_sad_labels \
            --channel 1 > exp/spectral_cluster/${partition}_${sad_type}_sad_rttm
fi
```

Convert the clustering labels into the Rich Transcription Time Marked (RTTM) format, saved in `exp/spectral_cluster/${partition}_${sad_type}_sad_rttm`.

RTTM files are space-delimited text files containing one turn per line, each line containing ten fields:

* `Type` -- segment type; should always by `SPEAKER`
* `File ID` -- file name; basename of the recording minus extension (e.g., `abjxc`)
* `Channel ID` -- channel (1-indexed) that turn is on; should always be `1`
* `Turn Onset` -- onset of turn in seconds from beginning of recording
* `Turn Duration` -- duration of turn in seconds
* `Orthography Field` -- should always by `<NA>`
* `Speaker Type` -- should always be `<NA>`
* `Speaker Name` -- name of speaker of turn; should be unique within scope of each file
* `Confidence Score` -- system confidence (probability) that information is correct; should always be `<NA>`
* `Signal Lookahead Time` -- should always be `<NA>`

For instance,

```
SPEAKER abjxc 1 0.400 6.640 <NA> <NA> 0 <NA> <NA>
SPEAKER abjxc 1 8.680 55.960 <NA> <NA> 0 <NA> <NA>
```


### Stage 8: Evaluate the Result (DER)

```
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
    ref_dir=data/voxconverse-master/
    #ref_dir=data/VoxSRC2023/voxconverse/
    echo -e "Get the DER results\n..."
    perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
         -c 0.25 \
         -r <(cat ${ref_dir}/${partition}/*.rttm) \
         -s exp/spectral_cluster/${partition}_${sad_type}_sad_rttm 2>&1 | tee exp/spectral_cluster/${partition}_${sad_type}_sad_res

    if [ ${get_each_file_res} -eq 1 ];then
        single_file_res_dir=exp/spectral_cluster/${partition}_${sad_type}_single_file_res
        mkdir -p $single_file_res_dir
        echo -e "\nGet the DER results for each file and the results will be stored underd ${single_file_res_dir}\n..."

        awk '{print $2}' exp/spectral_cluster/${partition}_${sad_type}_sad_rttm | sort -u  | while read file_name; do
            perl external_tools/SCTK-2.4.12/src/md-eval/md-eval.pl \
                 -c 0.25 \
                 -r <(cat ${ref_dir}/${partition}/${file_name}.rttm) \
                 -s <(grep "${file_name}" exp/spectral_cluster/${partition}_${sad_type}_sad_rttm) > ${single_file_res_dir}/${partition}_${file_name}_res
        done
        echo "Done!"
    fi
fi
```

Use the **SCTK** toolkit to compute the Diarization Error Rate (DER) metric, which is the sum of

* speaker error -- percentage of scored time for which the wrong speaker id is assigned within a speech region
* false alarm speech -- percentage of scored time for which a nonspeech region is incorrectly marked as containing speech
* missed speech -- percentage of scored time for which a speech region is incorrectly marked as not containing speech

For more details about DER, consult Section 6.1 of the [NIST RT-09 evaluation plan](https://web.archive.org/web/20100606092041if_/http://www.itl.nist.gov/iad/mig/tests/rt/2009/docs/rt09-meeting-eval-plan-v2.pdf).

The overall DER result would be saved in `exp/spectral_cluster/${partition}_${sad_type}_sad_res`.
Optionally, set `get_each_file_res` as `1` if you also want to get the DER result for each single file, which will be saved under dir `exp/spectral_cluster/${partition}_${sad_type}_single_file_res`.


