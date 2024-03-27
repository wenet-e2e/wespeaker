#!/bin/bash
#
#$ -cwd
#$ -V
#$ -N extract_embd
#$ -o extract_embd.out
#$ -e extract_embd.err
#$ -l gpu=4,ram_free=10G,mem_free=10G,core=2,matylda6=2,scratch=0.5,gpu_ram=16G
#$ -q long.q@@gpu
#export PATH="/mnt/matylda5/iplchot/python_public/anaconda3/bin:$PATH"
cd /mnt/matylda6/rohdin/expts/wespeaker/wespeaker_private/examples/sre/v3/
unset PYTHONPATH
unset PYTHOHOME

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
which python
export PATH=$PATH:/mnt/matylda6/rohdin/software/kaldi_20210625/tools/sph2pipe/

#./run.sh > logs/run.sh.log.extract.1 2>&1 
#./run.sh > logs/run.sh.log.extract.2 2>&1 
#./run.sh > logs/run.sh.log.extract.3 2>&1 
./run.sh > logs/run.sh.log.extract.4 2>&1 

