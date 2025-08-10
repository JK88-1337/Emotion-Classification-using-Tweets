#!/usr/bin/env python3
"""
Simple script to run training with the dair-ai/emotion dataset.
"""

import os
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and run the training script
from examples.train_with_dair_emotion import main

if __name__ == "__main__":
    print("Starting training with dair-ai/emotion dataset...")
    main() 