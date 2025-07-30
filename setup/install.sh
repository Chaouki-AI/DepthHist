#!/bin/bash

#install unzip and Parallel GNU
sudo apt-get install unzip -y
wget http://ftp.gnu.org/gnu/parallel/parallel-latest.tar.bz2
sudo tar xjf parallel*
rm para*.bz2
cd parallel-*
sudo ./configure && make
sudo make install
cd ..
rm -r par*

# Define the Anaconda download URL (you can adjust the version here if needed)
ANACONDA_URL="https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh"

# Define the installation directory (default is user's home directory)
INSTALL_DIR="$HOME/anaconda3"

# Step 1: Update system and install prerequisites (optional, depending on the system)
echo "Updating system and installing prerequisites..."
sudo apt-get update -y
sudo apt-get install curl -y


# Step 2: Download Anaconda installer
echo "Downloading Anaconda..."
curl -O $ANACONDA_URL

# Step 3: Run the installer
echo "Installing Anaconda..."
bash Anaconda3-2024.06-1-Linux-x86_64.sh -b -p $INSTALL_DIR

# Step 4: Initialize Anaconda (this adds Anaconda to PATH in .bashrc or .zshrc)
echo "Initializing Anaconda..."
$INSTALL_DIR/bin/conda init

# Step 5: Clean up the installer
echo "Cleaning up..."
rm Anaconda3-2024.06-1-Linux-x86_64.sh

# Step 6: Activate changes (you may need to restart the shell session for full effect)
echo "Installation complete. Please restart your terminal or run 'source ~/.bashrc' to use Anaconda."

# Load the changes to the shell
source ~/.bashrc

bash ./installEnv.sh
