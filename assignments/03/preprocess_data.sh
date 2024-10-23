#!/bin/bash
# -*- coding: utf-8 -*-

set -e

pwd=`dirname "$(readlink -f "$0")"`
base=$pwd/../..
src=fr
tgt=en
data=$base/data/$tgt-$src/

# change into base directory to ensure paths are valid
cd $base

# create preprocessed directory
mkdir -p $data/preprocessed/

# normalize and tokenize raw data
cat $data/raw/train.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q > $data/preprocessed/train.$src.p
cat $data/raw/train.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q > $data/preprocessed/train.$tgt.p

# train truecase models
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$src --corpus $data/preprocessed/train.$src.p
perl moses_scripts/train-truecaser.perl --model $data/preprocessed/tm.$tgt --corpus $data/preprocessed/train.$tgt.p

# apply truecase models to splits
cat $data/preprocessed/train.$src.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/train.$src 
cat $data/preprocessed/train.$tgt.p | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/train.$tgt

# prepare remaining splits with learned models
for split in valid test tiny_train
    do
        cat $data/raw/$split.$src | perl moses_scripts/normalize-punctuation.perl -l $src | perl moses_scripts/tokenizer.perl -l $src -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$src > $data/preprocessed/$split.$src
        cat $data/raw/$split.$tgt | perl moses_scripts/normalize-punctuation.perl -l $tgt | perl moses_scripts/tokenizer.perl -l $tgt -a -q | perl moses_scripts/truecase.perl --model $data/preprocessed/tm.$tgt > $data/preprocessed/$split.$tgt
    done

# Apply BPE dropout if specified
dropout_prob=$4
if [ ! -z "$dropout_prob" ]; then
    echo "Applying BPE dropout with probability: $dropout_prob"
    for split in train valid test tiny_train
    do
        python bpe_dropout.py --input $data/preprocessed/$split.$src \
                              --output $data/preprocessed/${split}_bpe.$src \
                              --dropout $dropout_prob
        python bpe_dropout.py --input $data/preprocessed/$split.$tgt \
                              --output $data/preprocessed/${split}_bpe.$tgt \
                              --dropout $dropout_prob
    done
fi

# preprocess all files for model training
if [ -z "$dropout_prob" ]; then
    # Baseline preprocessing
    python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/ --train-prefix $data/preprocessed/train --valid-prefix $data/preprocessed/valid --test-prefix $data/preprocessed/test --tiny-train-prefix $data/preprocessed/tiny_train --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
else
    # BPE Dropout preprocessing
    python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared_bpe/ --train-prefix $data/preprocessed/train_bpe --valid-prefix $data/preprocessed/valid_bpe --test-prefix $data/preprocessed/test_bpe --tiny-train-prefix $data/preprocessed/tiny_train_bpe --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
fi

echo "done."