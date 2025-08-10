#!/usr/bin/env python3
"""
Train feel-it models using the dair-ai/emotion dataset.

This script:
1. Loads the dair-ai/emotion dataset from Hugging Face
2. Adapts it to work with feel-it (maps emotions, handles language)
3. Trains custom sentiment and emotion models
4. Evaluates performance
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feel_it import SentimentClassifier, EmotionClassifier
from feel_it.trainer import SentimentTrainer, EmotionTrainer
from feel_it.custom_classifiers import CustomSentimentClassifier, CustomEmotionClassifier
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

def load_dair_emotion_dataset():
    """
    Load the dair-ai/emotion dataset from Hugging Face.
    """
    print("Loading dair-ai/emotion dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("dair-ai/emotion")
        
        # Convert to pandas DataFrames
        train_df = dataset['train'].to_pandas()
        validation_df = dataset['validation'].to_pandas()
        test_df = dataset['test'].to_pandas()
        
        print(f"Dataset loaded successfully!")
        print(f"Train samples: {len(train_df)}")
        print(f"Validation samples: {len(validation_df)}")
        print(f"Test samples: {len(test_df)}")
        
        return train_df, validation_df, test_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure you have the datasets library installed: pip install datasets")
        return None, None, None

def map_emotions_to_feelit(emotion_label):
    """
    Map dair-ai emotion labels to feel-it emotion labels.
    
    dair-ai emotions: joy, sadness, anger, fear, love, surprise
    feel-it emotions: anger, fear, joy, sadness
    """
    emotion_mapping = {
        0: 'sadness',    # sadness
        1: 'joy',        # joy  
        2: 'love',       # love -> map to joy
        3: 'anger',      # anger
        4: 'fear',       # fear
        5: 'surprise'    # surprise -> map to joy
    }
    
    emotion = emotion_mapping.get(emotion_label, 'joy')
    
    # Map love and surprise to joy since feel-it doesn't have these
    if emotion in ['love', 'surprise']:
        emotion = 'joy'
    
    return emotion

def create_sentiment_labels(emotion_label):
    """
    Create sentiment labels from emotion labels.
    """
    positive_emotions = ['joy', 'love', 'surprise']
    negative_emotions = ['sadness', 'anger', 'fear']
    
    emotion = map_emotions_to_feelit(emotion_label)
    
    if emotion in positive_emotions:
        return 'positive'
    else:
        return 'negative'

def prepare_dataset_for_feelit(train_df, validation_df, test_df):
    """
    Prepare the dair-ai dataset for feel-it training.
    """
    print("Preparing dataset for feel-it...")
    
    # Process training data
    train_data = []
    for _, row in train_df.iterrows():
        text = row['text']
        emotion_label = row['label']
        
        # Map to feel-it emotions
        emotion = map_emotions_to_feelit(emotion_label)
        sentiment = create_sentiment_labels(emotion_label)
        
        train_data.append({
            'text': text,
            'emotion': emotion,
            'sentiment': sentiment
        })
    
    # Process validation data
    val_data = []
    for _, row in validation_df.iterrows():
        text = row['text']
        emotion_label = row['label']
        
        emotion = map_emotions_to_feelit(emotion_label)
        sentiment = create_sentiment_labels(emotion_label)
        
        val_data.append({
            'text': text,
            'emotion': emotion,
            'sentiment': sentiment
        })
    
    # Process test data
    test_data = []
    for _, row in test_df.iterrows():
        text = row['text']
        emotion_label = row['label']
        
        emotion = map_emotions_to_feelit(emotion_label)
        sentiment = create_sentiment_labels(emotion_label)
        
        test_data.append({
            'text': text,
            'emotion': emotion,
            'sentiment': sentiment
        })
    
    # Convert to DataFrames
    train_processed = pd.DataFrame(train_data)
    val_processed = pd.DataFrame(val_data)
    test_processed = pd.DataFrame(test_data)
    
    print("Dataset preparation completed!")
    print(f"Training samples: {len(train_processed)}")
    print(f"Validation samples: {len(val_processed)}")
    print(f"Test samples: {len(test_processed)}")
    
    # Show label distributions
    print("\nEmotion distribution (training):")
    print(train_processed['emotion'].value_counts())
    print("\nSentiment distribution (training):")
    print(train_processed['sentiment'].value_counts())
    
    return train_processed, val_processed, test_processed

def create_custom_datasets(train_df, val_df, test_df):
    """
    Create custom datasets for training.
    """
    from feel_it.trainer import CustomTextDataset
    
    # Sentiment datasets
    sentiment_trainer = SentimentTrainer()
    sentiment_train_texts = train_df['text'].tolist()
    sentiment_train_labels = train_df['sentiment'].tolist()
    sentiment_val_texts = val_df['text'].tolist()
    sentiment_val_labels = val_df['sentiment'].tolist()
    
    # Tokenize
    sentiment_train_encodings = sentiment_trainer.tokenizer(
        sentiment_train_texts, truncation=True, padding=True, max_length=500
    )
    sentiment_val_encodings = sentiment_trainer.tokenizer(
        sentiment_val_texts, truncation=True, padding=True, max_length=500
    )
    
    # Convert labels to integers
    sentiment_train_numeric = [sentiment_trainer.reverse_sentiment_map[label] for label in sentiment_train_labels]
    sentiment_val_numeric = [sentiment_trainer.reverse_sentiment_map[label] for label in sentiment_val_labels]
    
    # Create datasets
    sentiment_train_dataset = CustomTextDataset(sentiment_train_encodings, sentiment_train_numeric)
    sentiment_val_dataset = CustomTextDataset(sentiment_val_encodings, sentiment_val_numeric)
    
    # Emotion datasets
    emotion_trainer = EmotionTrainer()
    emotion_train_texts = train_df['text'].tolist()
    emotion_train_labels = train_df['emotion'].tolist()
    emotion_val_texts = val_df['text'].tolist()
    emotion_val_labels = val_df['emotion'].tolist()
    
    # Tokenize
    emotion_train_encodings = emotion_trainer.tokenizer(
        emotion_train_texts, truncation=True, padding=True, max_length=500
    )
    emotion_val_encodings = emotion_trainer.tokenizer(
        emotion_val_texts, truncation=True, padding=True, max_length=500
    )
    
    # Convert labels to integers
    emotion_train_numeric = [emotion_trainer.reverse_emotion_map[label] for label in emotion_train_labels]
    emotion_val_numeric = [emotion_trainer.reverse_emotion_map[label] for label in emotion_val_labels]
    
    # Create datasets
    emotion_train_dataset = CustomTextDataset(emotion_train_encodings, emotion_train_numeric)
    emotion_val_dataset = CustomTextDataset(emotion_val_encodings, emotion_val_numeric)
    
    return (sentiment_train_dataset, sentiment_val_dataset, 
            emotion_train_dataset, emotion_val_dataset)

def train_models(train_df, val_df, test_df):
    """
    Train sentiment and emotion models.
    """
    print("\n=== Training Models ===")
    
    # Create datasets
    (sentiment_train_dataset, sentiment_val_dataset, 
     emotion_train_dataset, emotion_val_dataset) = create_custom_datasets(train_df, val_df, test_df)
    
    # Train sentiment model
    print("Training sentiment model...")
    sentiment_trainer = SentimentTrainer()
    sentiment_trainer.train(
        sentiment_train_dataset, 
        sentiment_val_dataset, 
        output_dir="./dair_sentiment_model",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    # Train emotion model
    print("Training emotion model...")
    emotion_trainer = EmotionTrainer()
    emotion_trainer.train(
        emotion_train_dataset, 
        emotion_val_dataset, 
        output_dir="./dair_emotion_model",
        num_epochs=3,
        batch_size=16,
        learning_rate=2e-5
    )
    
    print("Training completed!")

def evaluate_models(test_df):
    """
    Evaluate the trained models on test data.
    """
    print("\n=== Evaluating Models ===")
    
    # Load custom models
    sentiment_classifier = CustomSentimentClassifier("./dair_sentiment_model")
    emotion_classifier = CustomEmotionClassifier("./dair_emotion_model")
    
    # Load original models for comparison
    original_sentiment = SentimentClassifier()
    original_emotion = EmotionClassifier()
    
    # Prepare test data
    test_texts = test_df['text'].tolist()
    test_sentiment_labels = test_df['sentiment'].tolist()
    test_emotion_labels = test_df['emotion'].tolist()
    
    # Get predictions
    print("Getting predictions...")
    custom_sentiment_preds = sentiment_classifier.predict(test_texts)
    custom_emotion_preds = emotion_classifier.predict(test_texts)
    original_sentiment_preds = original_sentiment.predict(test_texts)
    original_emotion_preds = original_emotion.predict(test_texts)
    
    # Evaluate sentiment
    print("\n--- Sentiment Classification Results ---")
    custom_sentiment_acc = accuracy_score(test_sentiment_labels, custom_sentiment_preds)
    original_sentiment_acc = accuracy_score(test_sentiment_labels, original_sentiment_preds)
    
    print(f"Custom model accuracy: {custom_sentiment_acc:.3f}")
    print(f"Original model accuracy: {original_sentiment_acc:.3f}")
    
    print("\nCustom Sentiment Model Classification Report:")
    print(classification_report(test_sentiment_labels, custom_sentiment_preds))
    
    # Evaluate emotion
    print("\n--- Emotion Classification Results ---")
    custom_emotion_acc = accuracy_score(test_emotion_labels, custom_emotion_preds)
    original_emotion_acc = accuracy_score(test_emotion_labels, original_emotion_preds)
    
    print(f"Custom model accuracy: {custom_emotion_acc:.3f}")
    print(f"Original model accuracy: {original_emotion_acc:.3f}")
    
    print("\nCustom Emotion Model Classification Report:")
    print(classification_report(test_emotion_labels, custom_emotion_preds))
    
    return (custom_sentiment_preds, custom_emotion_preds, 
            original_sentiment_preds, original_emotion_preds)

def show_sample_predictions(test_df, custom_sentiment_preds, custom_emotion_preds, 
                          original_sentiment_preds, original_emotion_preds):
    """
    Show sample predictions for comparison.
    """
    print("\n=== Sample Predictions ===")
    
    # Show first 10 samples
    for i in range(min(10, len(test_df))):
        text = test_df.iloc[i]['text']
        true_sentiment = test_df.iloc[i]['sentiment']
        true_emotion = test_df.iloc[i]['emotion']
        
        print(f"\nText: {text[:100]}...")
        print(f"True - Sentiment: {true_sentiment}, Emotion: {true_emotion}")
        print(f"Custom - Sentiment: {custom_sentiment_preds[i]}, Emotion: {custom_emotion_preds[i]}")
        print(f"Original - Sentiment: {original_sentiment_preds[i]}, Emotion: {original_emotion_preds[i]}")

def main():
    """
    Main function to run the complete training pipeline.
    """
    print("=== Training Feel-It Models with dair-ai/emotion Dataset ===\n")
    
    # Load dataset
    train_df, validation_df, test_df = load_dair_emotion_dataset()
    if train_df is None:
        return
    
    # Prepare dataset
    train_processed, val_processed, test_processed = prepare_dataset_for_feelit(
        train_df, validation_df, test_df
    )
    
    # Train models
    train_models(train_processed, val_processed, test_processed)
    
    # Evaluate models
    (custom_sentiment_preds, custom_emotion_preds, 
     original_sentiment_preds, original_emotion_preds) = evaluate_models(test_processed)
    
    # Show sample predictions
    show_sample_predictions(test_processed, custom_sentiment_preds, custom_emotion_preds,
                          original_sentiment_preds, original_emotion_preds)
    
    print("\n=== Training Complete ===")
    print("Models saved to:")
    print("- ./dair_sentiment_model")
    print("- ./dair_emotion_model")
    print("\nYou can now use these models with CustomSentimentClassifier and CustomEmotionClassifier")

if __name__ == "__main__":
    main() 