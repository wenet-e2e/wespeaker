#!/bin/bash

# Copyright 2025 Your Name/Org (your_email@example.com)
# Adapted from run_wavlm.sh

. ./path.sh || exit 1

stage=1        # Start from stage 1 by default
stop_stage=-1

HOST_NODE_ADDR="localhost:29402"  # Keep or adjust as needed
num_nodes=1
job_id=2025   # Change if needed

# Data path (assuming same data as run_wavlm.sh)
data=data
data_type="shard"  # shard/raw

# --- Configuration for W2V-BERT ---
# Stage 1: Initial training (LoRA frozen encoder)
config_s1=conf/w2vbert_s1_lora.yaml
exp_dir_s1=exp/W2VBert_AdapterMFA_LoRA_frozen

# Stage 2: Joint Fine-tuning (full model)
config_s2=conf/w2vbert_s2_ft.yaml
exp_dir_s2=exp/W2VBert_AdapterMFA_joint_ft

# Stage 3: Large Margin Fine-tuning (full model)
config_s3=conf/w2vbert_s3_lmft.yaml
exp_dir_s3=exp/W2VBert_AdapterMFA_joint_lmft
# --- End Configuration ---

gpus="[0,1,2,3,4,5,6,7]"  # Adjust GPU list as needed
num_avg=1                 # Default averaging, adjust per stage if needed
checkpoint=

# Evaluation settings (keep or adjust)
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

. tools/parse_options.sh || exit 1

# Stage 1: Data preparation (assumed done by run_wavlm.sh or similar)
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Stage 1: Preparing datasets (assuming already done)..."
  # Optional: add data prep commands here if not done previously.
  # ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
  echo "Skipping data preparation, assuming it is done."
fi

# Stage 2: Data conversion (assumed done by run_wavlm.sh or similar)
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Stage 2: Converting train/test data to ${data_type} "\
"(assuming already done)..."
  # Optional: add conversion commands here if not done previously.
  # for dset in vox2_dev vox1; do
  #   if [ $data_type == "shard" ]; then ... else ... fi
  # done
  # python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
  echo "Skipping data conversion, assuming it is done."
fi

# Stage 3: initial training (LoRA with frozen encoder)
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Stage 3: Start initial LoRA training (encoder frozen)..."
  num_gpus=$(echo "$gpus" | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes="$num_nodes" --nproc_per_node="$num_gpus" \
    --rdzv_id="${job_id}_s1" --rdzv_backend="c10d" \
    --rdzv_endpoint="$HOST_NODE_ADDR" \
    wespeaker/bin/train.py --config "$config_s1" \
      --exp_dir "${exp_dir_s1}" \
      --gpus "$gpus" \
      --num_avg 1 \
      --data_type "${data_type}" \
      --train_data "${data}/vox2_dev/${data_type}.list" \
      --train_label "${data}/vox2_dev/utt2spk" \
      --reverb_data "${data}/rirs/lmdb" \
      --noise_data "${data}/musan/lmdb" \
      ${checkpoint:+--checkpoint "$checkpoint"}
      # Use checkpoint only if resuming S1
fi

# Intermediate evaluation after Stage 3 (optional but recommended)
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Stage 4: Evaluate LoRA frozen model..."
  avg_model_s1=${exp_dir_s1}/models/avg_model.pt
  num_avg_s1=$(yq '.num_avg' "$config_s1")
  echo "Averaging model for Stage 1..."
  python wespeaker/bin/average_model.py \
    --dst_model "$avg_model_s1" \
    --src_path "${exp_dir_s1}/models" \
    --num "${num_avg_s1:-1}"

  echo "Extracting embeddings for Stage 1 model..."
  local/extract_vox.sh \
    --exp_dir "$exp_dir_s1" --model_path "$avg_model_s1" \
    --nj 8 --gpus "$gpus" --data_type "$data_type" --data "${data}"

  echo "Scoring Stage 1 model..."
  local/score.sh \
    --stage 1 --stop-stage 2 --data "${data}" \
    --exp_dir "$exp_dir_s1" --trials "$trials"

  # Optional: score normalization and calibration for S1
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method "$score_norm_method" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s1" --trials "$trials"

  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method "$score_norm_method" \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s1" --trials "$trials"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  # Stage 5: merge LoRA weights
  echo "Stage 5: Merging LoRA weights"

  # Stage 1 directory, for example exp/W2VBert_AdapterMFA_LoRA_frozen
  S1_CONFIG_PATH=${exp_dir_s1}/config.yaml
  S1_CHECKPOINT_IN=${exp_dir_s1}/models/avg_model.pt
  S1_CHECKPOINT_OUT=${exp_dir_s1}/models/merged_avg_model.pt

  python tools/merge_lora.py \
    --config "${S1_CONFIG_PATH}" \
    --checkpoint_in "${S1_CHECKPOINT_IN}" \
    --checkpoint_out "${S1_CHECKPOINT_OUT}"
fi

# Stage 6: joint fine-tuning (full model)
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Stage 6: Start joint fine-tuning (full model)..."
  # Initialize S2 training from merged S1 model
  init_checkpoint_s2="${exp_dir_s1}/models/merged_avg_model.pt"
  if [ ! -f "$init_checkpoint_s2" ]; then
    echo "Error: Merged model from Stage 1 ($init_checkpoint_s2) "\
"not found."
    echo "Run previous stages."
    exit 1
  fi

  mkdir -p "${exp_dir_s2}/models"
  cp "$init_checkpoint_s2" "${exp_dir_s2}/models/model_0.pt"

  num_gpus=$(echo "$gpus" | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes="$num_nodes" --nproc_per_node="$num_gpus" \
    --rdzv_id="${job_id}_s2" --rdzv_backend="c10d" \
    --rdzv_endpoint="$HOST_NODE_ADDR" \
    wespeaker/bin/train.py --config "$config_s2" \
      --exp_dir "${exp_dir_s2}" \
      --gpus "$gpus" \
      --num_avg 1 \
      --data_type "${data_type}" \
      --train_data "${data}/vox2_dev/${data_type}.list" \
      --train_label "${data}/vox2_dev/utt2spk" \
      --reverb_data "${data}/rirs/lmdb" \
      --noise_data "${data}/musan/lmdb" \
      --checkpoint "${exp_dir_s2}/models/model_0.pt"
fi

# Intermediate evaluation after Stage 6
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Stage 7: Evaluate joint fine-tuned model..."
  avg_model_s2=${exp_dir_s2}/models/avg_model.pt
  num_avg_s2=$(yq '.num_avg' "$config_s2")
  echo "Averaging model for Stage 2..."
  python wespeaker/bin/average_model.py \
    --dst_model "$avg_model_s2" \
    --src_path "${exp_dir_s2}/models" \
    --num "${num_avg_s2:-1}"

  echo "Extracting embeddings for Stage 2 model..."
  local/extract_vox.sh \
    --exp_dir "$exp_dir_s2" --model_path "$avg_model_s2" \
    --nj 8 --gpus "$gpus" --data_type "$data_type" --data "${data}"

  echo "Scoring Stage 2 model..."
  local/score.sh \
    --stage 1 --stop-stage 2 --data "${data}" \
    --exp_dir "$exp_dir_s2" --trials "$trials"

  # Optional: score normalization and calibration for S2
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method "$score_norm_method" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s2" --trials "$trials"

  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method "$score_norm_method" \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s2" --trials "$trials"
fi

# Stage 8: large margin fine-tuning (full model)
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Stage 8: Start large margin fine-tuning (full model)..."
  # Initialize S3 training from averaged S2 model
  init_checkpoint_s3="${exp_dir_s2}/models/avg_model.pt"
  if [ ! -f "$init_checkpoint_s3" ]; then
    echo "Error: Averaged model from Stage 2 ($init_checkpoint_s3) "\
"not found."
    echo "Run previous stages."
    exit 1
  fi

  mkdir -p "${exp_dir_s3}/models"
  cp "$init_checkpoint_s3" "${exp_dir_s3}/models/model_0.pt"

  num_gpus=$(echo "$gpus" | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes="$num_nodes" --nproc_per_node="$num_gpus" \
    --rdzv_id="${job_id}_s3" --rdzv_backend="c10d" \
    --rdzv_endpoint="$HOST_NODE_ADDR" \
    wespeaker/bin/train.py --config "$config_s3" \
      --exp_dir "${exp_dir_s3}" \
      --gpus "$gpus" \
      --num_avg 1 \
      --data_type "${data_type}" \
      --train_data "${data}/vox2_dev/${data_type}.list" \
      --train_label "${data}/vox2_dev/utt2spk" \
      --reverb_data "${data}/rirs/lmdb" \
      --noise_data "${data}/musan/lmdb" \
      --checkpoint "${exp_dir_s3}/models/model_0.pt"
fi

# Final evaluation after Stage 8
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Stage 9: Evaluate large margin fine-tuned model..."
  avg_model_s3=${exp_dir_s3}/models/avg_model.pt
  num_avg_s3=$(yq '.num_avg' "$config_s3")
  echo "Averaging model for Stage 3..."
  python wespeaker/bin/average_model.py \
    --dst_model "$avg_model_s3" \
    --src_path "${exp_dir_s3}/models" \
    --num "${num_avg_s3:-1}"

  echo "Extracting embeddings for Stage 3 model..."
  local/extract_vox.sh \
    --exp_dir "$exp_dir_s3" --model_path "$avg_model_s3" \
    --nj 8 --gpus "$gpus" --data_type "$data_type" --data "${data}"

  echo "Scoring Stage 3 model..."
  local/score.sh \
    --stage 1 --stop-stage 2 --data "${data}" \
    --exp_dir "$exp_dir_s3" --trials "$trials"

  # Optional: score normalization and calibration for S3
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method "$score_norm_method" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s3" --trials "$trials"

  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method "$score_norm_method" \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev --top_n "$top_n" \
    --data "${data}" --exp_dir "$exp_dir_s3" --trials "$trials"
fi

echo "All stages completed."
