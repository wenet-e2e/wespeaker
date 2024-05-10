#!/bin/bash -l
#SBATCH --job-name=extract_MODEL_ID # Job name
#SBATCH --output=logs/wavlm/out_extract__MODEL_ID.%j     # Name of stdout output file
#SBATCH --error=logs/wavlm/err_extract__MODEL_ID.%j      # Name of stderr error file
#SBATCH --partition=small-g             # or ju-standard-g, partition name small-g
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=8          # 16 MPI ranks per node
#SBATCH --gpus-per-node=1             # Allocate one gpu per MPI rank
#SBATCH --mem=56GB
#SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000792   # Project for billing

# https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_2_06_Introduction_to_AMD_ROCm_Ecosystem/
# FIX: this fixed the error:
#      libtorch_cpu.so: undefined symbol: roctracer_next_record, version roctracer_4.1

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load gcc/11.2.0 

export PROJ_DIR="/scratch/project_465000792/xodehn09"
export DATA_DIR="${PROJ_DIR}/data"

SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2"
SCRIPT="$SCRIPT_DIR/run_evaluate.sh"


export NCCL_DEBUG=DEBUG

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214

# singularity exec --bind $PROJ_DIR:$PROJ_DIR --pwd $SCRIPT_DIR "$IMAGE" "$SCRIPT"

# exp_name="NAKI-only--WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch100-softmax"
# exp_name="NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-softmax"
exp_name=NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin

# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-no-margin
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-softmax

# exp_name=WavLM-BasePlus-Last_ASTP-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_Mean-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_PoolDim-emb257-3s-LRS10-Epoch20-no-margin

export exp_dir="exp/$exp_name"
export config="$exp_dir/conf.yaml"

export checkpoint="${exp_dir}/models/model_MODEL_ID.pt"
export eval_model=model_MODEL_ID.pt

# export checkpoint="${exp_dir}/models/model_20.pt"
# export eval_model=model_20.pt

export gpus="[0]"
export num_avg=2
export stage=3
export stop_stage=3

$SCRIPT


