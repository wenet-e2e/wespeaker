#!/bin/bash

# Copyright (c) 2023 Johan Rohdin (rohdin@fit.vutbr.cz)
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


set -o pipefail

export LC_ALL=C


sre18_dev_dir=""
sre18_eval_dir=""
sre18_eval_keys_file=""
data_dir=data/sre18
wav_dir=wav/
stage=1
stop_stage=1

. tools/parse_options.sh || exit 1

echo "sre18 dev dir: $sre18_dev_dir"
echo "sre18 eval dir: $sre18_eval_dir"
echo "sre18 eval keys file: $sre18_eval_keys_file"

declare -A set2dir=( ["dev"]=$sre18_dev_dir ["eval"]=$sre18_eval_dir )
declare -A set2subset=( ["dev"]="enrollment test unlabeled" ["eval"]="enrollment test" )


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then


    for z in dev eval;do
        src_dir=${set2dir[$z]}

        echo "Processing SRE 18 $z set from $src_dir"


        true && {
        for s in ${set2subset[$z]};do

            tgt_dir=$data_dir/$z/$s
            echo " - $s set. Storing in $tgt_dir"
            mkdir -p $tgt_dir

            if [ -f $tgt_dir/wav.scp ];then
                rm $tgt_dir/wav.scp
            fi

            # Create the wav files
            for x in $( ls $src_dir/data/${s}/ );do
                name=$(basename $x .sph)
                if [ $name != $x ];then
                    # suffix is .sph
                    echo "$name ffmpeg -nostdin -i $src_dir/data/${s}/$x -ar 8000 -f wav pipe:1 |" >> $tgt_dir/wav.scp
                else
                    name=$(basename $x .flac)
                    if [ $name != $x ];then
                        # suffix is .flac
                        # From http://trac.ffmpeg.org/wiki/audio%20types:"The default for muxing
                        # into WAV files is pcm_s16le." so the below should be ok.
                        echo "$name ffmpeg -nostdin -i $src_dir/data/${s}/$x -ar 8000 -f wav pipe:1 |" >> $tgt_dir/wav.scp
                    else
                        echo "ERROR: Invalid suffix in file $x"
                        exit 1
                    fi
                fi
            done
        done
        }

        # Mappings for "enrollment models" <-> "utterances"
        # The evaluation consider enrollment "models" rather than enrollment "speakers". Possibly several models could be
        # from the same speaker. There speaker ID of the models are not known. So we can't create "spk2utt" and utt2spk".
        # For test data there is no such mappings either.
        grep -v modelid $src_dir/docs/sre18_${z}_enrollment.tsv | cut -f1,2 | sed "s:\t: :" > $data_dir/$z/enrollment/enrollment.txt
        cat $data_dir/$z/enrollment/enrollment.txt | sed "s:.sph$::" |  sed "s:.flac$: :" \
            | awk '{print $2 " " $1}' > $data_dir/$z/enrollment/utt2mdl_id
        # No utterance is used in more than one mdl so utt2mdl_id makes sense.
        ./tools/utt2spk_to_spk2utt.pl $data_dir/$z/enrollment/utt2mdl_id >  $data_dir/$z/enrollment/mdl_id2utt

        true && {
        # Trial list and keys. Not available in the eval directory so the specified file is used.
        if [ $z == "eval" ];then
            cp $sre18_eval_keys_file $data_dir/$z/
            key_name=$( basename $sre18_eval_keys_file .tbz2)
            tar -xvf $data_dir/$z/${key_name}.tbz2 -C $data_dir/$z/
            key_file=$data_dir/$z/LDC2018E51/docs/sre18_eval_trial_key.tsv
        else
            key_file=$src_dir/docs/sre18_dev_trial_key.tsv
        fi
        }

        tail -n+2 $key_file | cut -f1,2,4 | sed "s:\.sph::" | sed "s:\.flac::" | sed "s:\t: :g" > $data_dir/$z/sre18_${z}_trials


    done
fi


