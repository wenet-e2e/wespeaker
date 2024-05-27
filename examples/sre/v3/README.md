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

