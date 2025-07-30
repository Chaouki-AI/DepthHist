#!/bin/bash
# This script was written by Mohammed Chaouki ZIARA, 
# For Ph.D. Project @ RCAM Laboratory, Djilali Liabes University - Algeria 
# Used to download the DIODE dataset (RGB images and depth maps) at maximum speed.
# Contact: medchaoukiziara@gmail.com || me@chaouki.pro

echo "🚀 STARTING HIGH-SPEED DOWNLOAD OF DIODE DATASET 🚀"
echo "Official Website: https://diode-dataset.org/"

# Get the directory where the script is located
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Create the 'data' directory in the script location
mkdir -p "$SCRIPT_DIR/data"

# URLs of the files to download
VAL_URL="http://diode-dataset.s3.amazonaws.com/val.tar.gz"
TRAIN_URL="http://diode-dataset.s3.amazonaws.com/train.tar.gz"

# Extract filenames from URLs
VAL_FILE="$SCRIPT_DIR/data/$(basename "$VAL_URL")"
TRAIN_FILE="$SCRIPT_DIR/data/$(basename "$TRAIN_URL")"

# Download both datasets in parallel using aria2c
aria2c -x 16 -s 16 -k 10M -d "$SCRIPT_DIR/data" -o "$(basename "$VAL_URL")" "$VAL_URL" &
aria2c -x 16 -s 16 -k 10M -d "$SCRIPT_DIR/data" -o "$(basename "$TRAIN_URL")" "$TRAIN_URL" &

wait  # Wait for both downloads to finish

# Verify if downloads are successful
if [[ -f "$VAL_FILE" && -s "$VAL_FILE" ]]; then
    echo "✅ Validation dataset downloaded successfully."
else
    echo "❌ Failed to download validation dataset."
fi

if [[ -f "$TRAIN_FILE" && -s "$TRAIN_FILE" ]]; then
    echo "✅ Training dataset downloaded successfully."
else
    echo "❌ Failed to download training dataset."
fi

echo "🎉 Download process completed!"
