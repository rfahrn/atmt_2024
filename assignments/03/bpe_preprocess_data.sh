#!/bin/bash
# -*- coding: utf-8 -*-

set -e
set -o pipefail
set -u

# Define base directories and language codes
current_dir="$(dirname "$(readlink -f "$0")")"
project_root="/mnt/c/Users/rebec/OneDrive/Desktop/Adv MT/atmt_2024"
src_lang="fr"
tgt_lang="en"
data_path="$project_root/data/en-fr"
moses_path="$project_root/moses_scripts"
prep_dir="$data_path/preprocessed_data"
ready_dir="$data_path/prepared_data"

# BPE and preprocessing settings
bpe_merges=5000          # Number of BPE merge operations
min_frequency=2           # Minimum frequency for vocabulary thresholding
dropout_prob=0.1          # Dropout probability during BPE application
vocab_size=10000          # Vocabulary size limit
min_len=1                 # Minimum sequence length
max_len=200               # Maximum sequence length

# Function to check if required tools are installed
check_tools() {
    local tools=("perl" "python3" "subword-nmt" "awk")
    echo "Checking for required tools..."
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            echo "Error: $tool is not installed."
            exit 1
        fi
    done
}

# Function to check if required Moses scripts are present
check_moses_scripts() {
    local scripts=(
        "$moses_path/normalize-punctuation.perl"
        "$moses_path/tokenizer.perl"
        "$moses_path/train-truecaser.perl"
        "$moses_path/truecase.perl"
    )
    echo "Checking for required Moses scripts..."
    for script in "${scripts[@]}"; do
        if [[ ! -f "$script" ]]; then
            echo "Error: $script not found."
            exit 1
        fi
    done
}

# Function to create necessary directories
create_directories() {
    local dirs=("$prep_dir" "$ready_dir" "$data_path/raw")
    echo "Creating directories..."
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
}

# Function to check if input files exist
verify_input_files() {
    local splits=("train" "valid" "test" "tiny_train")
    echo "Verifying input files..."
    for split in "${splits[@]}"; do
        for lang in "$src_lang" "$tgt_lang"; do
            local file="$data_path/raw/$split.$lang"
            if [[ ! -f "$file" ]]; then
                echo "Error: Input file not found: $file"
                exit 1
            fi
        done
    done
}

# Function to preprocess files
process_file() {
    local input_file=$1
    local output_file=$2
    local lang=$3
    local truecase_model=$4

    echo "Processing $input_file..."
    perl "$moses_path/normalize-punctuation.perl" -l "$lang" < "$input_file" | \
    perl "$moses_path/tokenizer.perl" -l "$lang" -a -q | \
    perl "$moses_path/truecase.perl" --model "$truecase_model" > "$output_file"
}

# Function to learn joint BPE codes
learn_joint_bpe_codes() {
    local src_file=$1
    local tgt_file=$2
    local codes_file="$prep_dir/bpe.codes"
    echo "Learning joint BPE codes..."
    cat "$src_file" "$tgt_file" > "$prep_dir/train_combined"
    subword-nmt learn-bpe -s $bpe_merges < "$prep_dir/train_combined" > "$codes_file"
    rm "$prep_dir/train_combined"
}

# Function to create vocabulary files
create_vocab() {
    local input_file=$1
    local vocab_file=$2
    echo "Creating vocabulary from $input_file..."
    subword-nmt get-vocab < "$input_file" > "$vocab_file"
}

# Function to apply BPE codes to files
apply_bpe() {
    local input_file=$1
    local output_file=$2
    local vocab_file=$3
    local dropout=$4
    local codes_file="$prep_dir/bpe.codes"

    echo "Applying BPE to $input_file..."
    if [[ -n "$dropout" ]]; then
        subword-nmt apply-bpe -c "$codes_file" --vocabulary "$vocab_file" --vocabulary-threshold 50 --dropout "$dropout" < "$input_file" > "$output_file"
    else
        subword-nmt apply-bpe -c "$codes_file" --vocabulary "$vocab_file" --vocabulary-threshold 50 < "$input_file" > "$output_file"
    fi
}

# Function to clean sequences based on length
clean_sequences() {
    local input_file=$1
    local output_file=$2
    echo "Cleaning sequences in $input_file..."
    awk -v min_len=$min_len -v max_len=$max_len '{
        num_words = NF
        if (num_words >= min_len && num_words <= max_len) {
            print $0
        }
    }' "$input_file" > "$output_file"
}

# Main preprocessing function
main() {
    # Check tools and scripts
    check_tools
    check_moses_scripts
    create_directories
    verify_input_files

    echo "Starting preprocessing..."

    # Step 1: Normalize, tokenize, and truecase training data
    echo "Processing training data..."
    for lang in "$src_lang" "$tgt_lang"; do
        input_file="$data_path/raw/train.$lang"
        temp_file="$prep_dir/train.tok.$lang"
        truecase_model="$prep_dir/truecase-model.$lang"
        output_file="$prep_dir/train.tc.$lang"

        # Normalize and tokenize
        echo "Normalizing and tokenizing $input_file..."
        perl "$moses_path/normalize-punctuation.perl" -l "$lang" < "$input_file" | \
        perl "$moses_path/tokenizer.perl" -l "$lang" -a -q > "$temp_file"

        # Train truecaser
        echo "Training truecaser for $lang..."
        perl "$moses_path/train-truecaser.perl" --model "$truecase_model" --corpus "$temp_file"

        # Apply truecaser
        echo "Applying truecaser to $temp_file..."
        perl "$moses_path/truecase.perl" --model "$truecase_model" < "$temp_file" > "$output_file"
    done

    # Step 2: Process validation and test data
    echo "Processing validation and test data..."
    for split in valid test tiny_train; do
        for lang in "$src_lang" "$tgt_lang"; do
            input_file="$data_path/raw/$split.$lang"
            output_file="$prep_dir/$split.tc.$lang"
            truecase_model="$prep_dir/truecase-model.$lang"

            if [[ -f "$input_file" ]]; then
                process_file "$input_file" "$output_file" "$lang" "$truecase_model"
            else
                echo "Warning: $input_file not found."
            fi
        done
    done

    # Step 3: Learn BPE codes
    echo "Learning BPE codes..."
    learn_joint_bpe_codes "$prep_dir/train.tc.$src_lang" "$prep_dir/train.tc.$tgt_lang"

    # Step 4: Create vocabularies
    echo "Creating vocabularies..."
    create_vocab "$prep_dir/train.tc.$src_lang" "$prep_dir/vocab.$src_lang"
    create_vocab "$prep_dir/train.tc.$tgt_lang" "$prep_dir/vocab.$tgt_lang"

    # Step 5: Apply BPE to training data
    echo "Applying BPE to training data..."
    for lang in "$src_lang" "$tgt_lang"; do
        input_file="$prep_dir/train.tc.$lang"
        output_file="$prep_dir/train.bpe.$lang"
        vocab_file="$prep_dir/vocab.$lang"
        clean_file="$prep_dir/train.clean.$lang"

        # Clean sequences
        clean_sequences "$input_file" "$clean_file"

        # Apply BPE with dropout for training data
        apply_bpe "$clean_file" "$output_file" "$vocab_file" "$dropout_prob"
    done

    # Step 6: Apply BPE to validation and test data
    echo "Applying BPE to validation and test data..."
    for split in valid test tiny_train; do
        for lang in "$src_lang" "$tgt_lang"; do
            input_file="$prep_dir/$split.tc.$lang"
            output_file="$prep_dir/$split.bpe.$lang"
            vocab_file="$prep_dir/vocab.$lang"
            clean_file="$prep_dir/$split.clean.$lang"

            # Clean sequences
            clean_sequences "$input_file" "$clean_file"

            # Apply BPE without dropout
            apply_bpe "$clean_file" "$output_file" "$vocab_file" ""
        done
    done

    # Step 7: Prepare data for training
    echo "Preparing data for training..."
    python3 "$project_root/preprocess.py"  \
        --target-lang "$tgt_lang" \
        --source-lang "$src_lang" \
        --dest-dir "$ready_dir" \
        --train-prefix "$prep_dir/train.bpe" \
        --valid-prefix "$prep_dir/valid.bpe" \
        --test-prefix "$prep_dir/test.bpe" \
        --tiny-train-prefix "$prep_dir/tiny_train.bpe" \
        --threshold-src "$min_frequency" \
        --threshold-tgt "$min_frequency" \
        --num-words-src "$vocab_size" \
        --num-words-tgt "$vocab_size"

    echo "Preprocessing completed successfully!"
}

# Run the main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi

