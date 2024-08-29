This is a **WeSpeaker** recipe for the Voxceleb 1&2 dataset. VoxCeleb is an audio-visual dataset consisting of short clips of human speech, extracted from interview videos uploaded to YouTube. See https://www.robots.ox.ac.uk/~vgg/data/voxceleb/ for more detailed information.

The following recipes are provided:
* v1: **Fully-Supervised** train on Voxceleb 1 development set and evaluate on Voxceleb1-O trials.

* v2: **Fully-Supervised** train on Voxceleb 2 development set and evaluate on three official trials.

* v2_deprecated: Deprecated version of fully-supervised train on Voxceleb dataset (deprecated IO).

* v3: **Self-Supervised** train on Voxceleb 2 development set and evaluate on three official trials, including SimCLR, MoCo and DINO.
