### Main differences from ../v2
* The training data is the CTS superset plus VoxCeleb with GSM codec
* The test data is SRE16, SRE18, and SRE21
* Preprocessing of embeddings before backend/scoring is supported

### Important
Similarly to ../v2, this recipe uses silero vad https://github.com/snakers4/silero-vad
downloaded from here https://github.com/snakers4/silero-vad/archive/refs/tags/v4.0.zip
If you intended to use this recipe for an evaluation/competition, make sure to check that
it is allowed to use the data that has been used to train Silero.

### Instructions
* Set the paths in stage 1. The variable ```sre_data_dir``` is assumed to be prepared by
  Kaldi (https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16/v2).
  Only the eval and unlabeled (major) data of sre16 is taken from there.
  ```voxceleb_dir``` is the path to voxceleb prepared by wespeaker (```../../voxceleb/v2```).
  If you set it to "" (empty string), the preparation will be run here. For the other datasets,
  the path to the folder provided by LDA should be provided. The relevant LDC numbers and
  file names of the data can be seen in the script. If you don't have
  one or two of the "eval/dev" sets of "sre16", "sre18" or "sre21" and not specify it, you may
  have to comment it from some more places in order to avoided crashes. (Eventually
  the script will hopefully be made more robust to this.)
  If you don't have the CTS superset data, you can skip stage 5 in ```local/prepare_data.sh```
  and instead replace the CTS data it with some other data, e.g., the training data prepared in ```../v2```
  If so, it is probably the easiest to name this data "CTS" since this name is assumed later
  in the recipe.
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
the necessary processing parameters (means, lda transforms etc.) and stores the whole processing chain with parameters in
pickle format. The parameters for each step will be calculated sequentially and the data specified for the parameter estimation of a step will
be processed by the  earlier steps. Therefore the data for the different steps can be different. For example when estimating LDA in the above chain, the data given by ```$lda_scp``` will first be processed by ```mean-subtract``` whose parameters were estimated by ```$mean1_scp``` which could be a different dataset.
In scenarios where unlabeled domain adaptation data is available, we want to use this data for the first mean subtraction while still using the out domain data for LDA estimation. This CANNOT be achieved by specifying the processing chain
```
mean-subtract --scp $indomain_scp | length-norm | lda --scp $lda_scp --utt2spk $utt2spk --dim $lda_dim | length-norm
```
since this would have the consequence that in LDA estimation, the data (```$lda_scp```) would be subjected to mean subtraction
using the mean of the indomain data (```$indomain_scp```). To solve this, we have an additional script ```wespeaker/bin/update_embd_proc.py``` used as follows
```
new_link="mean-subtract --scp $indomain_scp"
python wespeaker/bin/update_embd_proc.py --in_path $preprocessing_path_cts_aug --out_path $preprocessing_path_sre18_unlab --link_no_to_remove 0 --new_link "$new_link"
```
where ```$preprocessing_path_cts_aug``` is the path to the pickled original processing chain and ```$preprocessing_path_sre18_unlab``` is the path to the new pickled processing chain.
The script will remove link 0, e.g. ```mean-subtract --scp $mean1_scp``` and replace it with ```mean-subtract --scp $indomain_scp```.


### Regarding extractor training data pruning

Similarly to ```../v2``` and Kaldi's sre16 recipe, we discard some of the training utterances based on duration as well as training speakers based on their number of utterances.
This is controlled in stage 9 of ```local/prepare_data.sh```. It is quite flexible but currently a bit messy and some consequences of the settings are not obvious. Therefore some explanation is provided here.
There are three "blocks" in stage 9:
* The first block discards all utterances shorter or equal to some specified duration (currently set to 5s) according to VOICED DURATION.
* The second block discards all utterances shorter or equal to some specified duration (currently set to 5s) according to TOTAL DURATION, i.e., ignoring VAD info.
* The third block discards all speakers that has less than or equal to a specified number of utterances. (Currently set to 2, i.e. speaker with 3 or more utterances are kept.)
It is possible to set the thresholds differently for the different sets. IMPORTANT: The pruning in block 1 is based on ```data/data_set_name/utt2voice_dur``` which is calculated
from the VAD info, so if a recording does not have any speech, it will not be present in utt2voice_dur and therefore discarded in this block even if the duration threshold is
set to e.g. -1. If we want such utterances to be kept for one set we should not run this block for the set (as currently is the case for voxceleb). The current setup is as follows:
    1. Apply block one to CTS but not Voxceleb
    2. Apply block two to Voxceleb but not CTS. (Applying this stage to CTS would not have an effect if the thresholds are the same since the total duration is always larger or equal to the voiced duration.)
    3. Apply stage three to both CTS and VoxCeleb.

    This means Voxceleb recordings are kept even if they have no speech accordng to VAD. The later shard creation stage applies VAD if available, otherwise keeps the file as it is. So Voxceleb recording with no speech according to VAD will NOT be discarded (but there are only around 70 of them which is unlikely to have any effect on the trained system.). Also, there is a risk that pruning according to total duration while applying VAD in shard creation could result in recordings shorter than "num_frms". These will be zero padded at training time so there will be no crash but this is probably also suboptimal.
These is setting are arguably somewhat weird. Applying block one also to voxceleb (and not using block two at all) would be more reasonable but it seems to degrade the performance due to discarding too many files. A better solution than the current would be to try with smaller thresholds than 5s but we have had not had time to explore this yet. Also, it would be reasonable to discard recordings with no speech according to VAD in the shard creation stage. However, when no VAD is available for a file, the shard creation code does not know whether this is because no speech was detected for this file according to VAD, or because VAD was not ran for this file. Since we want to have the possibility to keep recordings for which the latter is the case, we have it this way (it could for example be considered not to use VAD for voxceleb at all, in which case we need to avoid discarding these files at the shard creation stage). A more flexible and clear solution is needed and we will work on this for future updates.


### Some data statistics
|                                              |  CTS #utt   | CTS #spk | CTS #utt | CTS #spk | comment|
| ---                                          |  ---    | ---  |    ---  |  --- |  --- |
|Original data                                 |  605760 | 6867 | 1245525 | 7245 |      |
|exclud recording with nospeech acording to VAD|  605704 | 6867 | 1245455 | 7245 | VAD is a bit random so these numbers could vary slightly, especially for voxceleb. |
|After filtering according voiced duration     |  604774 | 6867 |  816411 | 7245 | Accordingly, here too. We don't use this for voxceleb in the current settings.  |
|After filtering according total duration       |  -      | -    |  868326 | 7245 | Haven't checked this for CTS.

No speaker are discarded in block three with the current setting.


### Things to explore
Very few things have been tuned. For example the following could be low-hanging fruits:
* The above mentioned pruning rules
* Utterance durations of the training segments.
* Shall voxceleb be included? Is applying the GSM codec a good idea? (Note that GSM codec is applied in the data preparation stage while augmentation is applied at training time, i.e, GSM codec comes before augmentations. This is not so realistic, since in reality noise and reverberation comes before the data is recorded and encoded. However, it is consistent with CTS where we also apply augmentations at the already encoded audio since it was encoded at recording time.)
* The other architectures.

We will tune this futher in the future. We are also happy to hear about any such results obtained by others.