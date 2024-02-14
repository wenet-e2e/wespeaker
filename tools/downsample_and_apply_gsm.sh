#!/bin/bash
# split the wav scp, calculate duration and merge

src_dir=""
dest_dir=""
rate=""
wav_dir=""
remove_prefix_wav=""
nj=12

. tools/parse_options.sh || exit 1

if [ "A" == "A" ];then
    tools/copy_data_dir.sh $src_dir $dest_dir
    
    rm -f $dest_dir/wav_*.slice
    split --additional-suffix .slice -d -n l/$nj $dest_dir/wav.scp $dest_dir/wav_
    
    # Create the neceassary output directory structure.
    cut -d" " -f2 $dest_dir/wav.scp | xargs dirname | sed "s:$remove_prefix_wav:$wav_dir/:" | uniq | sort -u | xargs mkdir -p 


    rm $dest_dir/wav.scp
    for ((i=0; i<nj; i++));do
	{
	    n=$(printf %02d  $i)
	    slice=$dest_dir/wav_$n.slice

	    awk -v s=$remove_prefix_wav -v r=$rate -v w=$wav_dir '{o=$2; sub(s, w"/", $2); print "sox " o  " -t gsm -r " r " - | sox -t gsm -r " r " - -t wav -r " r " -c 1 -e signed-integer " $2    }' $slice > $dest_dir/sox_cmd_$n

	    bash $dest_dir/sox_cmd_$n
	} &
    done
    wait
fi


# Create the new scp
awk -v s=$remove_prefix_wav -v w=$wav_dir '{sub(s,w"/",$2); print $0 }' $src_dir/wav.scp > $dest_dir/wav.scp

# Some sanity checks
if [ $(wc -l $src_dir/wav.scp|cut -f1 -d" ") -ne $(find $wav_dir -name "*.wav"|wc -l|cut -f1 -d" ") ];then
    echo "ERROR: Not all data processed properly"
    exit 1
fi
if [ $(wc -l $src_dir/wav.scp|cut -f1 -d" ") -ne $(wc -l $dest_dir/wav.scp|cut -f1 -d" ") ];then
    echo "ERROR: New scp not produced correctly"
    exit 1
fi

# Clean up.
#rm -f $dest_dir/wav_*.slice
#rm -f $dest_dir/sox_cmd_*
