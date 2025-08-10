#!/usr/bin/env python3
"""
Script to only plot training curves from existing training history.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import json

def plot_training_curves(history, save_path="./training_curves.png"):
    """Plot accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Get the correct epoch ranges for each metric
    val_loss_epochs = range(1, len(history['emotion']['val_loss']) + 1)
    val_acc_epochs = range(1, len(history['emotion']['val_accuracy']) + 1)
    
    # For training loss, we need to map steps to epochs
    # Assuming 125 steps per epoch (1250 total steps / 10 epochs)
    steps_per_epoch = 125
    train_loss_epochs = []
    train_loss_values = []
    
    for i, loss in enumerate(history['emotion']['train_loss']):
        epoch = (i * 25) / steps_per_epoch  # 25 is logging_steps
        if epoch <= 10:  # Only plot up to 10 epochs
            train_loss_epochs.append(epoch)
            train_loss_values.append(loss)
    
    # Emotion Loss
    axes[0].plot(train_loss_epochs, train_loss_values, 'b-', label='Train Loss', alpha=0.7)
    axes[0].plot(val_loss_epochs, history['emotion']['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Emotion Model - Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_xlim(0, 10)  # Set x-axis to 0-10 epochs
    
    # Emotion Accuracy
    axes[1].plot(val_acc_epochs, history['emotion']['val_accuracy'], 'g-', label='Validation Accuracy')
    axes[1].set_title('Emotion Model - Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(0, 10)  # Set x-axis to 0-10 epochs
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training curves saved to: {save_path}")

def load_training_history_from_checkpoint(checkpoint_path):
    """Load training history from a checkpoint's trainer_state.json"""
    try:
        with open(os.path.join(checkpoint_path, 'trainer_state.json'), 'r') as f:
            trainer_state = json.load(f)
        
        # Extract training history from log_history
        log_history = trainer_state.get('log_history', [])
        
        # Reconstruct the training history structure
        history = {
            'emotion': {
                'train_loss': [],
                'val_loss': [],
                'val_accuracy': []
            }
        }
        
        for entry in log_history:
            if 'loss' in entry:
                history['emotion']['train_loss'].append(entry['loss'])
            if 'eval_loss' in entry:
                history['emotion']['val_loss'].append(entry['eval_loss'])
            if 'eval_accuracy' in entry:
                history['emotion']['val_accuracy'].append(entry['eval_accuracy'])
        
        return history
        
    except Exception as e:
        print(f"Error loading training history: {e}")
        return None

def main():
    """Load existing training history and plot curves."""
    
    # You can specify which checkpoint to load
    checkpoint_path = "./dair_emotion_model_10epochs/checkpoint-1250"  # Latest checkpoint
    
    print(f"Loading training history from: {checkpoint_path}")
    history = load_training_history_from_checkpoint(checkpoint_path)
    
    if history is None:
        print("Failed to load training history.")
        return
    
    print("Training history loaded successfully!")
    print(f"Training loss points: {len(history['emotion']['train_loss'])}")
    print(f"Validation loss points: {len(history['emotion']['val_loss'])}")
    print(f"Validation accuracy points: {len(history['emotion']['val_accuracy'])}")
    
    # Plot the training curves
    plot_training_curves(history)

if __name__ == "__main__":
    main()