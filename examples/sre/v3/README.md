Main differences from ../v2
* The training data is the CTS superset
* The test data is SRE16, SRE18, and SRE21
* Preprocessing of embeddings before backend/scoring is supported


Instructions
* Set the paths in stage 1. The sre16 is assumed to be prepared by 
  Kaldi (https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2).
  Only the eval and unlabeled (major) data is taken from here.
  voxceleb_dir is the path to voxceleb prepared by wespeaker (../../voxceleb/v2).
  If you set it to "", the preparation will be ran here. If you don't have
  any of the eval/dev sets sre16, sre18 or sre21 and not specify it, you may 
  have a comment it from some more places in order to avoided crashes. (Eventually
  the script will be made more robust to this.)
  If you don't have the CTS superset data, you can skip stage 5 in local/prepare_data.sh
  and instead replace cts it with some other data, e.g., the training data prepared in ../v2
  If so, it is probably the easiest to name this data "cts" since this name is assumed later 
  in the recipe.
  The relevant LDC numbers and file names of the data can be seen in the script. 
* Select which torchrun command to use in stage 3. The first line 
  (currently commented) is for "single-node, multi-worker" (one 
  pytorch job per machine). The second line is for "Stacked 
  single-node multi-worker" (more than one pytorch job may be 
  submitted to the same node in your cluster.) See  
  https://pytorch.org/docs/stable/elastic/run.html for explanations.
* Stage 3 (training) and stage 4 (embedding extraction) need GPU. You may have
  to arrange how to run these parts based on your environment.


Explanation of embedding processing:
The code supports flexible combinations of embedding processing steps, such as length-norm and LDA.
A processing chain is specified e.g., as follows
"mean-subtract --scp $mean1_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm"
The script wespeaker/bin/prep_embd_proc.py takes such a processing chain as input, loops through the processing steps (separated by "|"), calculates 
the necessary processing parameters (means, lda transforms etc) and stores the whole processing chain with parameters in 
pickle format. The parameters for each step will be calculated sequentially and the data specified for its parameter estimation will 
be processed by the  earlier steps. Note that the data for the different steps can be different. For example when estimating lda in the above chain, the data given by $lda_scp will first be processed by 
"mean-subtract" whose parameters were estimated by $mean1_scp which could be a different dataset.

In scenarios where unlabeled domain adaptation data is available, we want to use this data for the first mean subtraction while still using the out domain data for LDA estimation. This CANNOT be achieved by specifying the processing chain  
"mean-subtract --scp $indomain_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm"
since this would have the consequence that in lda estimation, the data ($lda_scp) would be subjected to mean subtraction
using the mean of the indomain data ($indomain_scp). To solve this, we and additional script wespeaker/bin/update_embd_proc.py used as follows
new_link="mean-subtract --scp $indomain_scp "
python wespeaker/bin/update_embd_proc.py --in_path $preprocessing_path_cts_aug --out_path $preprocessing_path_sre18_unlab --link_no_to_remove 0 --new_link "$new_link"
where $preprocessing_path_cts_aug is the path to the pickled original processing chain and $preprocessing_path_sre18_unlab is the path to the new pickled processing chain.
The script will remove link 0, e.g. "mean-subtract --scp $mean1_scp" and replace it with "mean-subtract --scp $indomain_scp ".