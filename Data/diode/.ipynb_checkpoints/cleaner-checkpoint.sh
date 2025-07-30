#!/bin/bash
# Optimized script for extracting and cleaning large datasets (KITTI, DIODE, etc.)
# Author: Mohammed Chaouki ZIARA
# Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

# ---------------------- SETUP ----------------------

# Get the script directory and set data folder
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
DATA_DIR="$SCRIPT_DIR/data"

# Ensure GNU parallel is installed
if ! command -v parallel &>/dev/null; then
    echo "Error: GNU Parallel is not installed. Install it with: sudo apt install parallel"
    exit 1
fi

# Ensure pigz (parallel gzip) is installed
if ! command -v pigz &>/dev/null; then
    echo "Error: pigz is not installed. Install it with: sudo apt install pigz"
    exit 1
fi

# ---------------------- FUNCTION DEFINITIONS ----------------------

# Function to extract a chunk of files from tar.gz in parallel
extract_chunk() {
    local tar_file=$1
    local folder=$2
    local file_list=$3
    
    echo "Extracting chunk from $tar_file..."
    mkdir -p "$folder"

    # Extract only the specified files
    tar -xzf "$tar_file" -C "$folder" --files-from="$file_list"
}

export -f extract_chunk

# Function to handle extraction efficiently
process_large_tar() {
    local tar_file=$1
    local folder=$2
    local num_chunks=8  # Adjust based on CPU cores

    if [ -f "$tar_file" ]; then
        echo "Processing $tar_file in parallel chunks..."

        mkdir -p "$folder"

        # Use pigz for multi-threaded decompression
        pigz -dc "$tar_file" | tar -xvC "$folder" &

        # List all files in the archive
        tar -tzf "$tar_file" > /tmp/file_list.txt

        # Split file list into smaller chunks for parallel extraction
        split -n l/$num_chunks /tmp/file_list.txt /tmp/chunk_

        # Run extraction in parallel
        parallel extract_chunk ::: "$tar_file" ::: "$folder" ::: /tmp/chunk_*

        # Cleanup temp files
        rm -f /tmp/file_list.txt /tmp/chunk_*

        # Remove the tar.gz file after extraction
        rm -f "$tar_file"
        echo "$tar_file extracted and removed."
    else
        echo "No tar.gz file found for $tar_file"
    fi
}

export -f process_large_tar

# ---------------------- PARALLEL EXECUTION ----------------------

# Run train and val extraction in parallel
parallel process_large_tar ::: "$DATA_DIR/train.tar.gz" "$DATA_DIR/val.tar.gz" ::: "$DATA_DIR/train" "$DATA_DIR/val"

# ---------------------- FINAL MESSAGE ----------------------

echo "All datasets extracted successfully!"
