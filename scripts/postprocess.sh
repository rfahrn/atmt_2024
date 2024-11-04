#!/usr/bin/env bash
# -*- coding: utf-8 -*-
set -e
# file for postprocessing - detokenization and detruecasing
infile=$1
outfile=$2
lang=$3
scripts=`dirname "$(readlink -f "$0")"`
base=$scripts/..
moses_scripts=$base/moses_scripts
# test data reference
reference=$base/data/en-fr/raw/test 
cat $infile | perl $moses_scripts/detruecase.perl | perl $moses_scripts/detokenizer.perl -q -l $lang > $outfile
cat $outfile | sacrebleu $reference.$lang > $base/assignments/03/BPE_BLEU_trans.txt