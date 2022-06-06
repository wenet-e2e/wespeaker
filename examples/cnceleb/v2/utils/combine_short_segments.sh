#!/usr/bin/env bash

# Copyright 2013  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script copies and modifies a data directory while combining
# segments whose duration is lower than a specified minimum segment
# length.
#
# Note: this does not work for the wav.scp, since there is no natural way to
# concatenate segments; you have to operate on directories that already have
# features extracted.

#


# begin configuration section
cleanup=true
speaker_only=false  # If true, utterances are only combined from the same speaker.
                    # It may be useful for the speaker recognition task.
                    # If false, utterances are preferentially combined from the same speaker,
                    # and then combined across different speakers.
get_dur_nj=20
# end configuration section


. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <min-segment-length-in-seconds> <dir>"
  echo "e.g.:"
  echo " $0 data/train 1.55 data/train_comb"
  echo " Options:"
  echo "  --speaker-only <true|false>  # options to internal/choose_utts_to_combine.py, default false."
  exit 1;
fi


export LC_ALL=C

srcdir=$1
min_seg_len=$2
dir=$3

if [ "$dir" == "$srcdir" ]; then
  echo "$0: this script requires <srcdir> and <dir> to be different."
  exit 1
fi

for f in $srcdir/utt2spk; do
  [ ! -s $f ] && echo "$0: expected file $f to exist and be nonempty" && exit 1
done

if ! mkdir -p $dir; then
  echo "$0: could not create directory $dir"
  exit 1;
fi

if ! utils/validate_data_dir.sh --no-text --no-feats --no-wav $srcdir; then
  echo "$0: failed to validate input directory $srcdir.  Run utils/fix_data_dir.sh $srcdir"
  utils/fix_data_dir.sh $srcdir
fi

if ! python -c "x=float('$min_seg_len'); assert(x>0.0 and x<100.0);" 2>/dev/null; then
  echo "$0: bad <min-segment-length-in-seconds>: got '$min_seg_len'"
  exit 1
fi

set -e
set -o pipefail

# make sure $srcdir/utt2dur exists.
if [ ! -f "$srcdir/utt2dur" ]; then
    utils/get_utt2dur.sh --nj $get_dur_nj $srcdir
fi

./utils/choose_utts_to_combine.py --min-duration=$min_seg_len \
  --merge-within-speakers-only=$speaker_only \
  $srcdir/spk2utt $srcdir/utt2dur $dir/utt2utts $dir/utt2spk $dir/utt2dur

utils/utt2spk_to_spk2utt.pl < $dir/utt2spk > $dir/spk2utt


