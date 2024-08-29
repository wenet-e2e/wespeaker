#!/usr/bin/perl
use warnings;        #sed replacement for -w perl parameter
use File::Basename;

# Copyright 2017   David Snyder
#           2024   Johan Rohdin (rohdin@fit.vutbr.cz)
# Apache 2.0
#

# This script is taken from the Kaldi SRE16 recipe. For the Wespeaker recipe, we
# have done a few very minor changes, namely:
# 1. The path to the keys tar file are provided as an additional input argument.
# 2. The produced wav.scp will use ffmpeg instead of sph2pipe.
# 3. Some changes in paths to fit wespeaker recipe.
# 4. Warning if wav files have no meta data. Mainly happens if the directory
#    searched for wav files contains files that are not in the original data.
# 5  Formatting to fit Wespeaker's requirement.

if (@ARGV != 3) {
  print STDERR "Usage: $0 <path-to-SRE16-eval> <path-to-sre16-evalset-keys>" ,
  " <path-to-output>\n";
  print STDERR "e.g. $0 /export/corpora/SRE/R149_0_1 data/\n";
  exit(1);
}

($db_base, $evalset_keys, $out_dir) = @ARGV;

# Handle enroll
$out_dir_enroll = "$out_dir/sre16/eval/enroll";
if (system("mkdir -p $out_dir_enroll")) {
  die "Error making directory $out_dir_enroll";
}

$tmp_dir_enroll = "$out_dir_enroll/tmp";
if (system("mkdir -p $tmp_dir_enroll") != 0) {
  die "Error making directory $tmp_dir_enroll";
}

open(SPKR, ">$out_dir_enroll/utt2spk")
    || die "Could not open the output file $out_dir_enroll/utt2spk";
open(WAV, ">$out_dir_enroll/wav.scp")
    || die "Could not open the output file $out_dir_enroll/wav.scp";
open(META, "<$db_base/docs/sre16_eval_enrollment.tsv")
    or die "cannot open wav list";
%utt2fixedutt = ();
while (<META>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];
  $utt = $toks[1];
  if ($utt ne "segment") {
    print SPKR "${spk}-${utt} $spk\n";
    $utt2fixedutt{$utt} = "${spk}-${utt}";
  }
}

# Using cmd here and a few other places to satisfy the 80 char. requirement.
my $cmd1="find $db_base/data/enrollment/ -name '*.sph'".
    " > $tmp_dir_enroll/sph.list";
if (system($cmd1) != 0) {
  die "Error getting list of sph files";
}

open(WAVLIST, "<$tmp_dir_enroll/sph.list") or die "cannot open wav list";

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$utt2fixedutt{$t1[0]};
  if ($utt) {
      print WAV "$utt",
      " ffmpeg -nostdin -i $sph -ac 1 -ar 8000 -f wav pipe:1 |\n";
  }else {
      print("WARNING $t1[0] not in meta data. Will not be used.\n");
  }
}
close(WAV) || die;
close(SPKR) || die;

# Handle test
$out_dir_test= "$out_dir/sre16/eval/test";
if (system("mkdir -p $out_dir_test")) {
  die "Error making directory $out_dir_test";
}

$tmp_dir_test = "$out_dir_test/tmp";
if (system("mkdir -p $tmp_dir_test") != 0) {
  die "Error making directory $tmp_dir_test";
}


if (system("cp $evalset_keys $out_dir_test")) {
    die "Error copying sre16 keys.";
}

my $key_name = basename( $evalset_keys );

if (system("tar -xvf $out_dir_test/$key_name -C $out_dir_test")) {
    die "Could not untar sre16 keys.";
}


open(SPKR, ">$out_dir_test/utt2spk")
    || die "Could not open the output file $out_dir_test/utt2spk";
open(WAV, ">$out_dir_test/wav.scp")
    || die "Could not open the output file $out_dir_test/wav.scp";
open(TRIALS, ">$out_dir_test/trials")
    || die "Could not open the output file $out_dir_test/trials";
open(TGL_TRIALS, ">$out_dir_test/trials_tgl")
    || die "Could not open the output file $out_dir_test/trials_tgl";
open(YUE_TRIALS, ">$out_dir_test/trials_yue")
    || die "Could not open the output file $out_dir_test/trials_yue";

my $cmd2="find $db_base/data/test/ -name '*.sph' > $tmp_dir_test/sph.list";
if (system($cmd2) != 0) {
  die "Error getting list of sph files";
}


open(KEY, "<$out_dir_test/R149_0_1/docs/sre16_eval_trial_key.tsv")
    || die "Could not open trials file",
    " $out_dir_test/R149_0_1/docs/sre16_eval_trial_key.tsv.";
open(SEG_KEY, "<$out_dir_test/R149_0_1/docs/sre16_eval_segment_key.tsv")
    || die "Could not open trials file",
    " $out_dir_test/R149_0_1/docs/sre16_eval_segment_key.tsv.";
open(LANG_KEY, "<$out_dir_test/R149_0_1/metadata/calls.tsv")
    || die " Could not open trials file",
    " $out_dir_test/R149_0_1/metadata/calls.tsv.";
open(WAVLIST, "<$tmp_dir_test/sph.list") or die "cannot open wav list";

%utt2call = ();
while(<SEG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $utt = $toks[0];
  $call = $toks[1];
  if ($utt ne "segment") {
    $utt2call{$utt} = $call;
  }
}
close(SEG_KEY) || die;

%call2lang = ();
while(<LANG_KEY>) {
  chomp;
  $line = $_;
  @toks = split(" ", $line);
  $call = $toks[0];
  $lang = $toks[1];
  $call2lang{$call} = $lang;
}
close(LANG_KEY) || die;

while(<WAVLIST>) {
  chomp;
  $sph = $_;
  @t = split("/",$sph);
  @t1 = split("[./]",$t[$#t]);
  $utt=$t1[0];
  print WAV "$utt"," ffmpeg -nostdin -i $sph -ac 1 -ar 8000 -f wav pipe:1 |\n";
  print SPKR "$utt $utt\n";
}
close(WAV) || die;
close(SPKR) || die;

while (<KEY>) {
  $line = $_;
  @toks = split(" ", $line);
  $spk = $toks[0];
  $utt = $toks[1];
  $call = $utt2call{$utt};
  $target_type = $toks[3];
  if ($utt ne "segment") {
    print TRIALS "${spk} ${utt} ${target_type}\n";
    if ($call2lang{$call} eq "tgl") {
      print TGL_TRIALS "${spk} ${utt} ${target_type}\n";
    } elsif ($call2lang{$call} eq "yue") {
      print YUE_TRIALS "${spk} ${utt} ${target_type}\n";
    } else {
      die "Unexpected language $call2lang{$call} for utterance $utt.";
    }
  }
}

close(TRIALS) || die;
close(TGL_TRIALS) || die;
close(YUE_TRIALS) || die;

my $cmd3="tools/utt2spk_to_spk2utt.pl".
    " $out_dir_enroll/utt2spk >$out_dir_enroll/spk2utt";
if (system($cmd3) != 0) {
  die "Error creating spk2utt file in directory $out_dir_enroll";
}

my $cmd4="tools/utt2spk_to_spk2utt.pl $out_dir_test/utt2spk >$out_dir_test/spk2utt";
if (system($cmd4) != 0) {
  die "Error creating spk2utt file in directory $out_dir_test";
}

if (system("tools/fix_data_dir.sh $out_dir_enroll") != 0) {
  die "Error fixing data dir $out_dir_enroll";
}
if (system("tools/fix_data_dir.sh $out_dir_test") != 0) {
  die "Error fixing data dir $out_dir_test";
}
