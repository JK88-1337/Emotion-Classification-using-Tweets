#!/usr/bin/env python3
"""
Streamlined training script with 30 epochs - generates accuracy/loss graphs only.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feel_it import EmotionClassifier
from feel_it.trainer import EmotionTrainer
from feel_it.custom_classifiers import CustomEmotionClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_local_dataset(data_folder="9444data", train_size=None, val_size=None, test_size=None):
    """Load dataset from local parquet files."""
    try:
        train_path = os.path.join(data_folder, "train-00000-of-00001.parquet")
        train_df = pd.read_parquet(train_path)
        if train_size:
            train_df = train_df.head(train_size)
        
        val_path = os.path.join(data_folder, "validation-00000-of-00001.parquet")
        val_df = pd.read_parquet(val_path)
        if val_size:
            val_df = val_df.head(val_size)
        
        test_path = os.path.join(data_folder, "test-00000-of-00001.parquet")
        test_df = pd.read_parquet(test_path)
        if test_size:
            test_df = test_df.head(test_size)
        
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None, None

def map_emotions_to_feelit(emotion_label):
    """Map dair-ai emotion labels to feel-it emotion labels."""
    emotion_mapping = {
        0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
    }
    emotion = emotion_mapping.get(emotion_label, 'joy')
    if emotion in ['love', 'surprise']:
        emotion = 'joy'
    return emotion

def prepare_dataset_for_feelit(train_df, validation_df, test_df):
    """Prepare the dair-ai dataset for feel-it training."""
    def process_data(df):
        data = []
        for _, row in df.iterrows():
            text = row['text']
            emotion_label = row['label']
            emotion = map_emotions_to_feelit(emotion_label)
            data.append({'text': text, 'emotion': emotion})
        return pd.DataFrame(data)
    
    train_processed = process_data(train_df)
    val_processed = process_data(validation_df)
    test_processed = process_data(test_df)
    
    return train_processed, val_processed, test_processed

def create_custom_datasets(train_df, val_df, test_df):
    """Create custom datasets for training."""
    from feel_it.trainer import CustomTextDataset
    
    # Emotion datasets
    emotion_trainer = EmotionTrainer()
    emotion_train_texts = train_df['text'].tolist()
    emotion_train_labels = train_df['emotion'].tolist()
    emotion_val_texts = val_df['text'].tolist()
    emotion_val_labels = val_df['emotion'].tolist()
    
    emotion_train_encodings = emotion_trainer.tokenizer(
        emotion_train_texts, truncation=True, padding=True, max_length=128
    )
    emotion_val_encodings = emotion_trainer.tokenizer(
        emotion_val_texts, truncation=True, padding=True, max_length=128
    )
    
    emotion_train_numeric = [emotion_trainer.reverse_emotion_map[label] for label in emotion_train_labels]
    emotion_val_numeric = [emotion_trainer.reverse_emotion_map[label] for label in emotion_val_labels]
    
    emotion_train_dataset = CustomTextDataset(emotion_train_encodings, emotion_train_numeric)
    emotion_val_dataset = CustomTextDataset(emotion_val_encodings, emotion_val_numeric)
    
    return emotion_train_dataset, emotion_val_dataset

def train_models_with_tracking(train_df, val_df, test_df, num_epochs=30, batch_size=8):
    """Train models with epoch-by-epoch tracking."""
    from transformers import TrainingArguments, Trainer
    from transformers.trainer_callback import TrainerCallback
    from sklearn.metrics import accuracy_score
    import torch
    import numpy as np
    
    # Create datasets
    emotion_train_dataset, emotion_val_dataset = create_custom_datasets(train_df, val_df, test_df)
    
    # Training history storage
    training_history = {
        'emotion': {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    }
    
    # Custom callback to track metrics
    class MetricsCallback(TrainerCallback):
        def __init__(self, history_key):
            self.history_key = history_key
            self.history = training_history[history_key]
        
        def on_evaluate(self, args, state, control, metrics, **kwargs):
            if 'eval_loss' in metrics:
                self.history['val_loss'].append(metrics['eval_loss'])
            if 'eval_accuracy' in metrics:
                self.history['val_accuracy'].append(metrics['eval_accuracy'])
        
        def on_log(self, args, state, control, logs, **kwargs):
            if 'loss' in logs:
                self.history['train_loss'].append(logs['loss'])
    
    # Compute metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {'eval_accuracy': accuracy_score(labels, predictions)}
    
    # Train emotion model
    print("Training emotion model...")
    emotion_trainer = EmotionTrainer()
    emotion_callback = MetricsCallback('emotion')
    
    emotion_training_args = TrainingArguments(
        output_dir="./dair_emotion_model_10epochs",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=50,  # Reduced for faster training
        weight_decay=0.01,
        logging_dir="./dair_emotion_model_10epochs/logs",
        logging_steps=25,  # More frequent logging
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=2e-5,  # Slightly higher for faster convergence
        report_to=None,  # Disable wandb/tensorboard
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        dataloader_num_workers=0,  # No parallel workers for CPU
    )
    
    emotion_trainer_obj = Trainer(
        model=emotion_trainer.model,
        args=emotion_training_args,
        train_dataset=emotion_train_dataset,
        eval_dataset=emotion_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[emotion_callback]
    )
    
    emotion_trainer_obj.train()
    emotion_trainer_obj.save_model()
    emotion_trainer.tokenizer.save_pretrained("./dair_emotion_model_10epochs")
    
    return training_history

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

def evaluate_final_models(test_df):
    """Evaluate the final trained models."""
    emotion_classifier = CustomEmotionClassifier("./dair_emotion_model_10epochs")
    
    test_texts = test_df['text'].tolist()
    test_emotion_labels = test_df['emotion'].tolist()
    
    emotion_preds = emotion_classifier.predict(test_texts)
    
    emotion_acc = accuracy_score(test_emotion_labels, emotion_preds)
    
    print(f"\nFinal Test Results:")
    print(f"Emotion Accuracy: {emotion_acc:.3f}")

def main():
    """Main training function."""
    print("=== Training Emotion Model with 10 Epochs (CPU Optimized) ===")
    
    # Training parameters (optimized for CPU speed)
    TRAIN_SIZE = 1000
    VAL_SIZE = 200
    TEST_SIZE = 200
    NUM_EPOCHS = 10
    BATCH_SIZE = 8  # Small batch size for CPU
    
    print(f"Training for {NUM_EPOCHS} epochs...")
    
    # Load dataset
    train_df, validation_df, test_df = load_local_dataset(
        data_folder="9444data",
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE
    )
    
    if train_df is None:
        print("Failed to load dataset.")
        return
    
    # Prepare dataset
    train_processed, val_processed, test_processed = prepare_dataset_for_feelit(
        train_df, validation_df, test_df
    )
    
    # Train models with tracking
    history = train_models_with_tracking(train_processed, val_processed, test_processed, 
                                       num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    # Plot training curves
    plot_training_curves(history)
    
    # Evaluate final models
    evaluate_final_models(test_processed)
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main() 