#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Usage example:
#   (atmt311) bash preprocess_data.sh data/en-fr en fr  
#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# Usage example:
#   bash preprocess_data.sh /path/to/data src_lang tgt_lang
set -e

base=$(dirname "$(readlink -f "$0")")/../..
data=$1           # Data directory passed as the first argument
src_lang=$2       # Source language passed as the second argument
tgt_lang=$3       # Target language passed as the third argument

cd "$base" || exit

mkdir -p "$data/preprocessed/"
mkdir -p "$data/prepared/BPE"
echo "preprocessing data for $src_lang-$tgt_lang"

cat "$data/raw/train.$src_lang" | perl moses_scripts/normalize-punctuation.perl -l $src_lang | perl moses_scripts/tokenizer.perl -l $src_lang -a -q > "$data/preprocessed/train.$src_lang.p"
cat "$data/raw/train.$tgt_lang" | perl moses_scripts/normalize-punctuation.perl -l $tgt_lang | perl moses_scripts/tokenizer.perl -l $tgt_lang -a -q > "$data/preprocessed/train.$tgt_lang.p"
echo "normalizing and tokenizing done!"

perl moses_scripts/train-truecaser.perl --model "$data/preprocessed/tm.$src_lang" --corpus "$data/preprocessed/train.$src_lang.p"
perl moses_scripts/train-truecaser.perl --model "$data/preprocessed/tm.$tgt_lang" --corpus "$data/preprocessed/train.$tgt_lang.p"

cat "$data/preprocessed/train.$src_lang.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$src_lang" > "$data/preprocessed/train.$src_lang"
cat "$data/preprocessed/train.$tgt_lang.p" | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$tgt_lang" > "$data/preprocessed/train.$tgt_lang"
echo "truecasing done!"

for split in valid test tiny_train
do
    cat "$data/raw/$split.$src_lang" | perl moses_scripts/normalize-punctuation.perl -l $src_lang | perl moses_scripts/tokenizer.perl -l $src_lang -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$src_lang" > "$data/preprocessed/$split.$src_lang"
    cat "$data/raw/$split.$tgt_lang" | perl moses_scripts/normalize-punctuation.perl -l $tgt_lang | perl moses_scripts/tokenizer.perl -l $tgt_lang -a -q | perl moses_scripts/truecase.perl --model "$data/preprocessed/tm.$tgt_lang" > "$data/preprocessed/$split.$tgt_lang"
done
echo "preprocessing done!"

rm "$data/preprocessed/train.$src_lang.p"
rm "$data/preprocessed/train.$tgt_lang.p"
echo "cleaning up..."

echo "learning BPE... using subword-nmt"

#C:/Users/rebec/miniconda3/envs/atmt311/Scripts/subword-nmt learn-bpe --input "$data/preprocessed/train.$src_lang" --output "$data/prepared/BPE/codes.$src_lang" --symbols 10000
#C:/Users/rebec/miniconda3/envs/atmt311/Scripts/subword-nmt learn-bpe --input "$data/preprocessed/train.$tgt_lang" --output "$data/prepared/BPE/codes.$tgt_lang" --symbols 10000

#C:/Users/rebec/miniconda3/envs/atmt311/Scripts/subword-nmt get-vocab --input "$data/preprocessed/train.$src_lang" --output "$data/prepared/BPE/dict.$src_lang"
#C:/Users/rebec/miniconda3/envs/atmt311/Scripts/subword-nmt get-vocab --input "$data/preprocessed/train.$tgt_lang" --output "$data/prepared/BPE/dict.$tgt_lang"
#echo "BPE learning done!"
#source activate atmt311
python -m subword_nmt.learn_bpe --input "$data/preprocessed/train.$src_lang" --output "$data/prepared/BPE/codes.$src_lang" --symbols 10000
python -m subword_nmt.learn_bpe --input "$data/preprocessed/train.$tgt_lang" --output "$data/prepared/BPE/codes.$tgt_lang" --symbols 10000

python -m subword_nmt.get_vocab --input "$data/preprocessed/train.$src_lang" --output "$data/prepared/BPE/dict.$src_lang"
python -m subword_nmt.get_vocab --input "$data/preprocessed/train.$tgt_lang" --output "$data/prepared/BPE/dict.$tgt_lang"

# For apply-bpe calls, also use python -m
for split in train tiny_train test valid
do
  python -m subword_nmt.apply_bpe --input "$data/preprocessed/$split.$src_lang" --codes "$data/prepared/BPE/codes.$src_lang" --output "$data/prepared/BPE/$split.$src_lang" --vocabulary "$data/prepared/BPE/dict.$src_lang"
  python -m subword_nmt.apply_bpe --input "$data/preprocessed/$split.$tgt_lang" --codes "$data/prepared/BPE/codes.$tgt_lang" --output "$data/prepared/BPE/$split.$tgt_lang" --vocabulary "$data/prepared/BPE/dict.$tgt_lang"
done


python preprocess.py --target-lang $tgt --source-lang $src --dest-dir $data/prepared/BPE --train-prefix $data/prepared/BPE/train --valid-prefix $data/prepared/BPE/valid --test-prefix $data/prepared/BPE/test --tiny-train-prefix $data/prepared/BPE/tiny_train --threshold-src 12 --threshold-tgt 12 --num-words-src 3000 --num-words-tgt 3000 --vocab-src $data/prepared/BPE/dict.fr --vocab-trg $data/prepared/BPE/dict.en

echo "done!"