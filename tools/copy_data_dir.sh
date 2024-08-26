#!/usr/bin/env bash

# Copyright           2023  Brno University of Techology (author: Johan Rohdin)
# Apache 2.0

# Copies wav.scp as well as utt2spk spk2utt if they are available. The script
# can also take a list of speakers or utterances to keep. If provided, only
# utterances/speakers in the list are kept.

src_dir=$1
dest_dir=$2

shift 2

update_wav_path=false
utt_list=""
spk_list=""

. tools/parse_options.sh || exit 1

if [ "$dest_dir" == "$src_dir" ]; then
  echo "$0 ERROR: Input directory (<src_dir>) and output directory (<dest_dir>) are the same."
  exit 1
fi

mkdir -p $dest_dir


if [ ! -z "$utt_list" ]; then
    echo "UTTLIST"
fi
if [ ! -z "$spk_list" ]; then
    echo "SPKLIST"
fi




#if [ $utt_list != "" ] && [ $spk_list != "" ]; then
if [ ! -z "$utt_list" ] && [ ! -z "$spk_list" ]; then
  echo "$0 ERROR: Providing both utt_list and spk_list not supported."
  exit 1
fi



if [ ! -f $src_dir/utt2spk ]; then
    echo "$0 WARNING: copy_data_dir.sh: no such file $src_dir/utt2spk"
else
    if [ ! -z "$utt_list" ];then
        awk 'NR==FNR{a[$1];next}$1 in a{print $0}' $utt_list $src_dir/utt2spk > $dest_dir/utt2spk
    elif [ ! -z "$spk_list" ];then
        #echo "A"
        awk 'NR==FNR{a[$1];next}$2 in a{print $0}' $spk_list $src_dir/utt2spk > $dest_dir/utt2spk
    else
        cp $src_dir/utt2spk $dest_dir/utt2spk
    fi
fi


if [ ! -f $src_dir/spk2utt ]; then
    echo "$0 WARNING: copy_data_dir.sh: no such file $src_dir/spk2utt"
else
    if [ ! -z "$utt_list" ];then
        # This will work even if utt2spk doesn't exist and was simpler than reducing spk2utt directly.
        cat $scrdir/spk2utt | tools/spk2utt_to_utt2spk.pl \
            | awk 'NR==FNR{a[$1];next}$1 in a{print $0}' $utt_list - \
            | tools/utt2spk_to_spk2utt.pl > $dest_dir/spk2utt

    elif [ ! -z "$spk_list" ];then
        awk 'NR==FNR{a[$1];next}$1 in a{print $0}' $spk_list $src_dir/spk2utt > $dest_dir/spk2utt
    else
        cp $src_dir/spk2utt $dest_dir/spk2utt
    fi
fi


if [ ! -f $src_dir/wav.scp ]; then
    echo "$0 ERROR: copy_data_dir.sh: no such file $src_dir/wav.scp"
    exit 1;
else
    if [ $update_wav_path == true ];then
        src_root_dir=$(readlink -f $src_dir | sed "s:data/.*::")
        dest_root_dir=$(readlink -f $dest_dir | sed "s:data/.*::")
        cat $src_dir/wav.scp | sed "s:$src_root_dir:$dest_root_dir:" > $dest_dir/wav.scp
    else
        cp $src_dir/wav.scp $dest_dir/wav.scp
    fi
fi


# Sanity checks
if [ -f $dest_dir/utt2spk ];then
    if [ $( wc -l $dest_dir/utt2spk | cut -f1 -d" ") -ne $( wc -l $dest_dir/wav.scp | cut -f1 -d" " ) ];then
        echo "ERROR: Length of utt2spk and wav.scp doesn't match."
        exit 1
    fi
    if [ -f $src_dir/spk2utt ]; then
        if [ $( cat $dest_dir/utt2spk | sort | md5sum | cut -f1 -d" " ) != $( tools/spk2utt_to_utt2spk.pl $dest_dir/spk2utt | sort | md5sum | cut -f1 -d" " ) ];then
            echo "ERROR: utt2spk and spk2utt doesn't match."
            exit 1
        fi
    fi
fi




