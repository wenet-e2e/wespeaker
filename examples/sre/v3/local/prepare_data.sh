#!/bin/bash

# Copyright (c) 2023 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2024 Johan Rohdin (rohdin@fit.vutbr.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

stage=-1
stop_stage=-1
#sre_data_dir=
data=data

###
sre16_unlab_dir=""
sre16_evalset_dir=""
sre16_evalset_keys=""
###
sre18_devset_dir=""
sre18_evalset_dir=""
sre18_evalset_keys=""
###
sre21_devset_dir=""
sre21_evalset_dir=""
sre21_evalset_keys=""
###
cts_superset_dir=""
###
voxceleb_dir=""

compute_total_utterance_duration=true # Whether to compute the total utterance duration, i.e., including no speech parts
                                      # Can be used as an addition filtering requirement. Currently only supported for
                                      # VoxCeleb.
compute_vad_for_voxceleb=true
include_voxceleb_vad_in_train_data=true # If false, only CTS vad will be inluded which means that VAD will not be applied for VoxCeleb during training.

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    mkdir -p external_tools
    # Download voice activity detection model pretrained by Silero Team
    wget -c https://github.com/snakers4/silero-vad/archive/refs/tags/v4.0.zip -O external_tools/silero-vad-v4.0.zip
    unzip -o external_tools/silero-vad-v4.0.zip -d external_tools
fi


### SRE16
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # We use the scripts from the Kaldi SRE16 recipe with some minor modifications.

    # Prepare NIST SRE 2016 evaluation data.
    local/make_sre16_eval.pl $sre16_evalset_dir $sre16_evalset_keys data

    # Prepare unlabeled Cantonese and Tagalog development data. This dataset
    # was distributed to SRE participants.
    local/make_sre16_unlabeled.pl $sre16_unlab_dir data
fi


### SRE18
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "Preparing SRE18"
    local/prepare_sre18.sh --stage 1 --stop_stage 1 --sre18_dev_dir $sre18_devset_dir --sre18_eval_dir $sre18_evalset_dir --sre18_eval_keys_file $sre18_evalset_keys --data_dir $data/sre18
fi


### SRE21
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "Preparing SRE21"
    local/prepare_sre21.sh --stage 1 --stop_stage 1 --sre21_dev_dir $sre21_devset_dir --sre21_eval_dir $sre21_evalset_dir --sre21_eval_keys_file $sre21_evalset_keys --data_dir $data/sre21
fi


### CTS
if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "Preparing CTS"
    local/prepare_cts_superset.sh --cts_superset_dir $cts_superset_dir --data_cts $data/cts --wav_dir `pwd`/wav/cts


    # Only mixer data. Used for backend training. Create only lists here.
    # The data directory will be created later, after VAD.
    awk -F"\t" '{if($7 == "mx3" || $7 ==  "mx45" || $7 == "mx6"){print $0}  }' ${cts_superset_dir}/docs/cts_superset_segment_key.tsv \
         > data/cts_superset_segment_key_mx3456.tsv
    cut -f 1  data/cts_superset_segment_key_mx3456.tsv | sed s:\\.sph$:: > data/mx_3456.list

fi


### VoxCeleb
# We are using all of VoxCeleb 1 and the training (aka "development") part of VoxCeleb 2.
# (The test part of VoxCeleb 2) may have some overlap with VoxCeleb 1. See
# https://www.robots.ox.ac.uk/~vgg/publications/2019/Nagrani19/nagrani19.pdf, Table 4.)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then

    echo "Preparing VoxCeleb"
    if [[ $voxceleb_dir == "" ]];then
        echo "Preparing Voxceleb, rirs and Musan"
        voxceleb_dir=${data}_vox
        mkdir ${voxceleb_dir}
        local/prepare_vox.sh --stage 1 --stop_stage 4 --data ${data}_vox
    fi

    if [[ ! -d $voxceleb_dir/vox1  ||  ! -d $voxceleb_dir/vox2_dev ]];then
        echo "ERROR: Problem with Voxceleb data directory."
        exit 1
    fi

    # Downsample VoxCeleb and apply GSM. We create a new wav.scp with this command in the
    # extraction chain rather than creating the new wav files explicitly.
    sox_command='-t gsm -r 8000 - | sox -t gsm -r 8000 - -t wav -r 8000 -c 1 -e signed-integer -'
    for dset in vox1 vox2_dev;do
        tools/copy_data_dir.sh $voxceleb_dir/$dset $data/${dset}_gsmfr
        awk -v sc="$sox_command" '{print $1 " sox " $2 " " sc " |" }' $voxceleb_dir/$dset/wav.scp > $data/${dset}_gsmfr/wav.scp
    done

    # Combine all Voxceleb data
    tools/combine_data.sh data/vox_gsmfr data/vox1_gsmfr/ data/vox2_dev_gsmfr/

    # Copy rirs and musan from voxceleb. We don't need to downsample as this will be
    # done on-the-fly. If the direcotires already contain the data in lmdb format
    # we just link it. Otherwise we copy it and let later stages create the lmdb
    # format data here. Since we don't want to affect the original data.
    for x in rirs musan;do
        if [ -d $voxceleb_dir/$x/lmdb ];then
            ln -s $voxceleb_dir/$x $data/
        else
            mkdir $data/$x
            cp -r $voxceleb_dir/$x/wav.scp $data/$x/wav.scp
        fi
    done

fi


if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then

    echo "Get vad segmentation for dataset."
    true && {
        # Set VAD min duration
        min_duration=0.25
        for dset in vox_gsmfr cts sre18/dev/test sre18/dev/enrollment sre18/dev/unlabeled sre18/eval/test sre18/eval/enrollment sre21/dev/test sre21/dev/enrollment sre21/eval/test sre21/eval/enrollment sre16_major sre16/eval/enrollment sre16/eval/test; do
            python3 local/make_system_sad.py \
                --repo-path external_tools/silero-vad-4.0 \
                --scp ${data}/${dset}/wav.scp \
                --min-duration $min_duration > ${data}/${dset}/vad
            cp -r  ${data}/${dset} ${data}/${dset}-bk # Since VAD is quite time-consuming, it is good to have a backup.
        done
    }

    true && {
        # We may consider to use only the mixer portion of the CTS data for backen training
        # as it may be closer to the SRE data.

        tools/subset_data_dir.sh --utt-list data/mx_3456.list data/cts data/mx_3456
        tools/filter_scp.pl -f 2 ${data}/mx_3456/wav.scp ${data}/cts/vad > ${data}/mx_3456/vad


        # For PLDA training, it is better to augment the training data
        python3 local/generate_sre_aug.py --ori_dir ${data}/mx_3456 \
            --aug_dir ${data}/mx_3456_aug \
            --aug_copy_num 2

        tools/utt2spk_to_spk2utt.pl ${data}/mx_3456_aug/utt2spk > ${data}/mx_3456_aug/spk2utt
    }

    true && {
        # We may consider to use only the mixer portion of the CTS data for backend training
        # as it may be closer to the SRE data.

        # For PLDA training, it is better to augment the training data
        python3 local/generate_sre_aug.py --ori_dir ${data}/cts \
            --aug_dir ${data}/cts_aug \
            --aug_copy_num 2

        tools/utt2spk_to_spk2utt.pl ${data}/cts_aug/utt2spk > ${data}/cts_aug/spk2utt
    }

fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then

    true && {
    for dset in cts vox_gsmfr; do
        echo $dset
        if [ -f ${data}/${dset}/vad ] && ( [ $dset != "vox_gsmfr" ] || $compute_vad_for_voxceleb ) ;then
            echo "Using VAD info"
            python3 local/utt2voice_duration.py \
                 --vad_file ${data}/${dset}/vad \
                 --utt2voice_dur ${data}/${dset}/utt2voice_dur
            cp ${data}/${dset}/utt2voice_dur ${data}/${dset}-bk/             # Good to have backup also of this
        fi
    done
    }

    true && {
    # The below need to be improved to work for a general wav.scp. It only works for the specif format of voxceleb wav.scp
    # at the moment.
    for dset in vox_gsmfr; do
        if $compute_total_utterance_duration; then
            # We may, for example, avoid applying VAD on VoxCeleb in which case we need this.
            # Note that the durations are estimated on the original wave file, before sox
            # downsampling and GSM codec is applied.
            echo "Using soxi"

            cut -f3 -d" " ${data}/${dset}/wav.scp | awk '{ print "soxi -D " $0 }' > ${data}/${dset}/soxi_cmd.sh
            split -a 4 -d -n l/12 ${data}/${dset}/soxi_cmd.sh ${data}/${dset}/soxi_cmd.split.
            for i in {0000..11}; do
                    bash ${data}/${dset}/soxi_cmd.split.$i > ${data}/${dset}/soxi_cmd.split.$i.out &
            done
            wait

            for i in {0000..11}; do cat ${data}/${dset}/soxi_cmd.split.$i.out; done > ${data}/${dset}/dur_tmp
            cut -f1 -d" " ${data}/${dset}/wav.scp > ${data}/${dset}/utt_tmp
            paste -d " " ${data}/${dset}/utt_tmp ${data}/${dset}/dur_tmp > ${data}/${dset}/utt2dur

            rm ${data}/${dset}/soxi_cmd.* ${data}/${dset}/vox_gsmfr/dur_tmp ${data}/${dset}/utt_tmp

            cp ${data}/${dset}/utt2dur ${data}/${dset}-bk/             # Good to have backup also of this
        fi
    done
    }
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then

    declare -A voice_dur_threshold=( ["cts"]=5.0 ["vox_gsmfr"]=0.0 ) # Note that a threshold of 0.0 still means that utterances with no speech
                                                                     # according to VAD will be discarded at this stage. So if we want to keep
                                                                     # them, we should skip block 1 for the set instead.
    declare -A dur_threshold=( ["cts"]=0.0 ["vox_gsmfr"]=5.0 )

    declare -A uttPerSpk_threshold=( ["cts"]=2 ["vox_gsmfr"]=2 )     # Kept if more than this threshold. (I.e. equality not sufficient.)

    true && {
    # Following the Kaldi recipe: https://github.com/kaldi-asr/kaldi/blob/71f38e62cad01c3078555bfe78d0f3a527422d75/egs/sre16/v2/run.sh#L189
    # We filter out the utterances with duration less than 5s
    echo "Stage 9, block 1"
    echo "Applying filtering based on voice duration "
    #for dset in cts vox_gsmfr; do
    for dset in cts; do
        n_utt_before=$( wc -l ${data}/${dset}/utt2spk | cut -f1 -d " " )
        n_spk_before=$( wc -l ${data}/${dset}/spk2utt | cut -f1 -d " " )
        python3 local/filter_utt_accd_dur.py \
            --wav_scp ${data}/${dset}/wav.scp \
            --utt2voice_dur ${data}/${dset}/utt2voice_dur \
            --filter_wav_scp ${data}/${dset}/filter_wav.scp \
            --dur_thres ${voice_dur_threshold[$dset]}
        mv ${data}/${dset}/wav.scp ${data}/${dset}/wav.scp.bak
        mv ${data}/${dset}/filter_wav.scp ${data}/${dset}/wav.scp
        tools/fix_data_dir.sh ${data}/${dset}
        echo " $dset "
        echo " #utt / #spk before: $n_utt_before / $n_spk_before "
        n_utt_after=$( wc -l ${data}/${dset}/utt2spk | cut -f1 -d " " )
        n_spk_after=$( wc -l ${data}/${dset}/spk2utt | cut -f1 -d " " )
        echo " #utt / #spk after: $n_utt_after / $n_spk_after "
    done
    }
    echo "Stage 9, block 2"
    echo "Applying filtering based on the whole utterance duration (including non-speech parts) "
    #for dset in cts vox_gsmfr; do
    for dset in vox_gsmfr; do
        n_utt_before=$( wc -l ${data}/${dset}/utt2spk | cut -f1 -d " " )
        n_spk_before=$( wc -l ${data}/${dset}/spk2utt | cut -f1 -d " " )
        python3 local/filter_utt_accd_dur.py \
            --wav_scp ${data}/${dset}/wav.scp \
            --utt2voice_dur ${data}/${dset}/utt2dur \
            --filter_wav_scp ${data}/${dset}/filter_wav.scp \
            --dur_thres ${dur_threshold[$dset]}
        mv ${data}/${dset}/wav.scp ${data}/${dset}/wav.scp.bak
        mv ${data}/${dset}/filter_wav.scp ${data}/${dset}/wav.scp
        tools/fix_data_dir.sh ${data}/${dset}
        echo " $dset "
        echo " #utt / #spk before: $n_utt_before / $n_spk_before "
        n_utt_after=$( wc -l ${data}/${dset}/utt2spk | cut -f1 -d " " )
        n_spk_after=$( wc -l ${data}/${dset}/spk2utt | cut -f1 -d " " )
        echo " #utt / #spk after: $n_utt_after / $n_spk_after "
    done


    # Similarly, following the Kaldi recipe,
    # we throw out speakers with fewer than 3 utterances.
    echo "Stage 9, block 3"
    for dset in cts vox_gsmfr; do
        #tools/fix_data_dir.sh ${data}/${dset}
        cp ${data}/${dset}/spk2utt ${data}/${dset}/spk2utt.bak
        awk -v thr=${uttPerSpk_threshold[$dset]} '{if(NF>thr){print $0}}' ${data}/${dset}/spk2utt.bak > ${data}/${dset}/spk2utt
        tools/spk2utt_to_utt2spk.pl ${data}/${dset}/spk2utt > ${data}/${dset}/utt2spk
        tools/fix_data_dir.sh ${data}/${dset}
    done

    ./tools/combine_data.sh data/cts_vox data/cts/ data/vox_gsmfr
    if $include_voxceleb_vad_in_train_data;then
        cat data/cts/vad data/vox_gsmfr/vad > data/cts_vox/vad
    else
        cat data/cts/vad > data/cts_vox/vad
    fi
fi


