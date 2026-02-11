# Define your data directory
DATA_DIR="/home/yifa/xiyang/wespeaker/dataset"

# 1. Generate wav.scp (format: <utt_id> <absolute_path>)
find $DATA_DIR -name "*.wav" | awk -F'/' '{print $(NF) " " $0}' | sed 's/.wav//1' > all_wav.scp

# 2. Generate utt2spk (format: <utt_id> <speaker_id>)
# This assumes the parent folder name of the wav file is the Speaker ID
awk '{split($1, a, "_"); print $1 " " a[1]}' all_wav.scp > all_utt2spk

# 3. Sort them (Important for Kaldi/WeSpeaker tools)
sort -u all_wav.scp -o all_wav.scp
sort -u all_utt2spk -o all_utt2spk

# Get total line count
total_lines=$(wc -l < all_wav.scp)
train_lines=$((total_lines * 80 / 100))

# Randomize and split
shuf all_wav.scp > all_wav_shuffled.scp
head -n $train_lines all_wav_shuffled.scp > train_wav.scp
tail -n +$((train_lines + 1)) all_wav_shuffled.scp > dev_wav.scp

# Sync the utt2spk files based on the split wav files
grep -Ff <(awk '{print $1}' train_wav.scp) all_utt2spk > train_utt2spk
grep -Ff <(awk '{print $1}' dev_wav.scp) all_utt2spk > dev_utt2spk