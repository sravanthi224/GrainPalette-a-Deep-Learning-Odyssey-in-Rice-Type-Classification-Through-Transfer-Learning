#!/usr/bin/env bash

# Exit on error
set -o errexit

# Install required system-level packages (especially libGL for OpenCV)
apt-get update && apt-get install -y libgl1-mesa-glx

# Install Python packages
pip install -r rice-classifier/requirements.txt
