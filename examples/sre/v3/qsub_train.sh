#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N train_xvec
#$ -o train_xvec.out
#$ -e train_xvec.err
#$ -pe smp 16
#$ -l gpu=0.125,ram_free=1.25G,mem_free=1.25G,matylda6=0.625,gpu_ram=16G
#$ -q long.q@@gpu
cd /mnt/matylda6/rohdin/expts/wespeaker/wespeaker_private_test2/examples/sre/v3/ # Need to change your training directory.

unset PYTHONPATH
unset PYTHONHOME

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/mnt/matylda5/iplchot/python_public/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/matylda5/iplchot/python_public/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/matylda5/iplchot/python_public/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="$PATH:/mnt/matylda5/iplchot/python_public/anaconda3/bin"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate /mnt/matylda6/rohdin/conda/wespeaker_20240220/
./run.sh > logs/run.sh.stage3.log.1 2>&1


