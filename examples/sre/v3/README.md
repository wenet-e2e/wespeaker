### Main differences from ../v2
* The training data is the CTS superset plus voxceleb with GSM codec
* The test data is SRE16, SRE18, and SRE21
* Preprocessing of embeddings before backend/scoring is supported


### Instructions
* Set the paths in stage 1. The sre16 is assumed to be prepared by 
  Kaldi (https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2).
  Only the eval and unlabeled (major) data is taken from here.
  voxceleb_dir is the path to voxceleb prepared by wespeaker (```../../voxceleb/v2```).
  If you set it to "" (empty string), the preparation will be run here. If you don't have
  any of the "eval/dev" sets of "sre16", "sre18" or "sre21" and not specify it, you may 
  have a comment it from some more places in order to avoided crashes. (Eventually
  the script will be made more robust to this.)
  If you don't have the CTS superset data, you can skip stage 5 in ```local/prepare_data.sh```
  and instead replace cts it with some other data, e.g., the training data prepared in ```../v2```
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


### Explanation of embedding processing

The code supports flexible combinations of embedding processing steps, such as length-norm and LDA.
A processing chain is specified e.g., as follows
```
mean-subtract --scp $mean1_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm"
```
The script ```wespeaker/bin/prep_embd_proc.py``` takes such a processing chain as input, loops through the processing steps (separated by ```|```), calculates 
the necessary processing parameters (means, lda transforms etc) and stores the whole processing chain with parameters in 
pickle format. The parameters for each step will be calculated sequentially and the data specified for its parameter estimation will 
be processed by the  earlier steps. Note that the data for the different steps can be different. For example when estimating lda in the above chain, the data given by ```$lda_scp``` will first be processed by ```mean-subtract``` whose parameters were estimated by ```$mean1_scp``` which could be a different dataset.
In scenarios where unlabeled domain adaptation data is available, we want to use this data for the first mean subtraction while still using the out domain data for LDA estimation. This CANNOT be achieved by specifying the processing chain
```  
mean-subtract --scp $indomain_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm
```
since this would have the consequence that in lda estimation, the data (```$lda_scp```) would be subjected to mean subtraction
using the mean of the indomain data (```$indomain_scp```). To solve this, we and additional script ```wespeaker/bin/update_embd_proc.py``` used as follows
```
new_link="mean-subtract --scp $indomain_scp "
python wespeaker/bin/update_embd_proc.py --in_path $preprocessing_path_cts_aug --out_path $preprocessing_path_sre18_unlab --link_no_to_remove 0 --new_link "$new_link"
```
where ```$preprocessing_path_cts_aug``` is the path to the pickled original processing chain and ```$preprocessing_path_sre18_unlab``` is the path to the new pickled processing chain.
The script will remove link 0, e.g. ```mean-subtract --scp $mean1_scp``` and replace it with ```mean-subtract --scp $indomain_scp```.


### Regarding extractor training data pruning

Similarly to ../v2 and Kaldi's sre16 recipe, we discard some of the training utterance based on duration as well as training speakers based on their number of utterances. 
This is controlled in stage 9 of local/prepare_data.sh. It is quite flexible but it is currently a bit messy and some consequences of the settings are not obvious. Therefore some explanation is provided here. 
There are three "blocks" in stage 9: 
* The first block discards all utterances shorter or equal to some specified duration (currently set to 5s.) according to VOICED DURATION. 
* The second block discards all utterances shorter or equal to some specified duration (currently set to 5s.) according to TOTAL DURATION, i.e., ignoring VAD info.
* The third block discards all speakers that has less than or equal to a specified number of utterances. (Currently set to 2, i.e. speaker with 3 or more utterances are kept.) 
It is possible to set the thresholds differently for the different sets. IMPORTANT: The pruning in block 1 is based on data/"data_set_name"/utt2voice_dur which is calculated 
from the VAD info, so if a recording does not have any speech, it will not be present in utt2voice_dur and therefore discarded in this block even the duration threshold is 
set to e.g. -1. If we want such utterances to be kept for one set we should not run this block for the set. The current setup is as follows:
* Apply block one to CTS but not Voxceleb
* Apply block two to Voxceleb but not CTS. (Applying this stage to CTS would not have an effect if the thresholds are the same since the total duration is always larger or equal to the voiced duration.
* Apply stage three to both CTS and VoxCeleb.)
This means Voxceleb recordings are kept even if there are no speech in them accordng to VAD. On the other hand, VAD is applied in the shard creation stage if available for both CTS and VoxCeleb while recording with no VAD info will be kept as is here so Voxceleb recording with no speech according to VAD will not be discarded (but there are only around 70 of them which is unlikely to have any effect on the trained system.). Also, there is a risk that pruning only according to total duration while applying VAD in shard creation could result in recordings shorter than "num_frms". These will be zero padded so there will be no crash, still this is probably suboptimal.
These is setting are arguably somewhat weird. Applying block one also to voxceleb (and not using block two att all) would be more reasonable but it seems to degrade the performance due to discarding too many files. A better solution than the current would be to try with smaller thresholds than 5s but we have had not had time to explore this yet. Also, it would be reasonable to discard recordings with no speech according to VAD at the shard creation stage. However, when no vad is available for a file, the shard creation code does not know whether this is because no speech was detected for this file according to VAD, or because VAD was not ran for this file. And we want to have the possibility to keep recordings for which the latter is the case (it would for example be reasonable not to use VAD for voxceleb at all, in which case we need to avoid discarding these files at the shard cretion stage). A more flexible solution is needed here and we will work on this as well as cleaner and simpler settings in this part in future updates.


### Some data statistics
|                                              |  CTS #utt   | CTS #spk | CTS #utt | CTS #spk | comment|
| ---                                          |  ---    | ---  |    ---  |  --- |  --- |
|Origanal data                                 |  605760 | 6867 | 1245525 | 7245 |      |
|exclud recording with nospeech acording to VAD|  605704 | 6867 | 1245455 | 7245 | VAD is a bit random so this number could vary slightly, especially for voxceleb |
|After filtering according voiced duration     |  604774 | 6867 |  816411 | 7245 | Accordingly, here too. We don't use this for voxceleb in the current settings.  |
|After filtering according tota duration       |  -      | -    |  868326 | 7245 | Haven't checked this for CTS.                   

No speaker are discarded in block three with the current setting.
   

### Things to explore
Very few things have been tuned. For example the following could be low-hanging fruits:
* The above mentioned pruning rules
* Utterance durations of the training segments. 
* Shall voxceleb be included? Is applying the GSM codec a good idea? (Note that GSM codec is applied in the data preparation stage while augmentation is applied at training time, i.e, GSM codec comes before augmentations since in reality, noise and reverberation comes before the data is recorded and encoded. This is not so realistica but consistent with CTS where we also apply augmentation at the already recorded audio.)
* The other architectures. 

We tune this futher in the future. We are also happy to hear about any such results obtained by others. 



