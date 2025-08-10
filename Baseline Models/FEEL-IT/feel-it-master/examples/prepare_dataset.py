#!/usr/bin/env python3
"""
Dataset preparation utility for feel-it training.

This script helps you prepare your dataset in the correct format for training
sentiment and emotion classification models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os

def validate_sentiment_labels(labels):
    """
    Validate that sentiment labels are in the correct format.
    
    Args:
        labels: List of labels
    
    Returns:
        valid_labels: List of validated labels
    """
    valid_sentiments = ['positive', 'negative']
    valid_labels = []
    
    for label in labels:
        if isinstance(label, str):
            label_lower = label.lower().strip()
            if label_lower in valid_sentiments:
                valid_labels.append(label_lower)
            else:
                print(f"Warning: Invalid sentiment label '{label}'. Using 'negative' as default.")
                valid_labels.append('negative')
        else:
            print(f"Warning: Non-string label '{label}'. Using 'negative' as default.")
            valid_labels.append('negative')
    
    return valid_labels

def validate_emotion_labels(labels):
    """
    Validate that emotion labels are in the correct format.
    
    Args:
        labels: List of labels
    
    Returns:
        valid_labels: List of validated labels
    """
    valid_emotions = ['anger', 'fear', 'joy', 'sadness']
    valid_labels = []
    
    for label in labels:
        if isinstance(label, str):
            label_lower = label.lower().strip()
            if label_lower in valid_emotions:
                valid_labels.append(label_lower)
            else:
                print(f"Warning: Invalid emotion label '{label}'. Using 'anger' as default.")
                valid_labels.append('anger')
        else:
            print(f"Warning: Non-string label '{label}'. Using 'anger' as default.")
            valid_labels.append('anger')
    
    return valid_labels

def clean_text(texts):
    """
    Basic text cleaning.
    
    Args:
        texts: List of text strings
    
    Returns:
        cleaned_texts: List of cleaned text strings
    """
    cleaned_texts = []
    
    for text in texts:
        if isinstance(text, str):
            # Remove extra whitespace
            cleaned = ' '.join(text.split())
            # Remove empty strings
            if cleaned.strip():
                cleaned_texts.append(cleaned)
            else:
                print(f"Warning: Empty text found, skipping.")
        else:
            print(f"Warning: Non-string text '{text}', skipping.")
    
    return cleaned_texts

def prepare_sentiment_dataset(input_file, text_column, label_column, output_file="sentiment_dataset.csv"):
    """
    Prepare a sentiment classification dataset.
    
    Args:
        input_file: Path to input CSV file
        text_column: Name of text column
        label_column: Name of label column
        output_file: Path to output CSV file
    """
    print(f"Preparing sentiment dataset from {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Extract columns
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Clean and validate
    texts = clean_text(texts)
    labels = validate_sentiment_labels(labels)
    
    # Ensure same length
    min_length = min(len(texts), len(labels))
    texts = texts[:min_length]
    labels = labels[:min_length]
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'text': texts,
        'sentiment': labels
    })
    
    # Save to file
    output_df.to_csv(output_file, index=False)
    print(f"Sentiment dataset saved to {output_file}")
    print(f"Total samples: {len(output_df)}")
    print(f"Label distribution:\n{output_df['sentiment'].value_counts()}")
    
    return output_df

def prepare_emotion_dataset(input_file, text_column, label_column, output_file="emotion_dataset.csv"):
    """
    Prepare an emotion classification dataset.
    
    Args:
        input_file: Path to input CSV file
        text_column: Name of text column
        label_column: Name of label column
        output_file: Path to output CSV file
    """
    print(f"Preparing emotion dataset from {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Extract columns
    texts = df[text_column].tolist()
    labels = df[label_column].tolist()
    
    # Clean and validate
    texts = clean_text(texts)
    labels = validate_emotion_labels(labels)
    
    # Ensure same length
    min_length = min(len(texts), len(labels))
    texts = texts[:min_length]
    labels = labels[:min_length]
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'text': texts,
        'emotion': labels
    })
    
    # Save to file
    output_df.to_csv(output_file, index=False)
    print(f"Emotion dataset saved to {output_file}")
    print(f"Total samples: {len(output_df)}")
    print(f"Label distribution:\n{output_df['emotion'].value_counts()}")
    
    return output_df

def split_dataset(input_file, output_dir=".", test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    
    Args:
        input_file: Path to input CSV file
        output_dir: Directory to save split datasets
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    print(f"Splitting dataset {input_file}...")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Determine label column
    if 'sentiment' in df.columns:
        label_column = 'sentiment'
        task_type = 'sentiment'
    elif 'emotion' in df.columns:
        label_column = 'emotion'
        task_type = 'emotion'
    else:
        raise ValueError("No 'sentiment' or 'emotion' column found in dataset")
    
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[label_column]
    )
    
    # Save splits
    train_file = os.path.join(output_dir, f"{task_type}_train.csv")
    test_file = os.path.join(output_dir, f"{task_type}_test.csv")
    
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    print(f"Train set saved to {train_file} ({len(train_df)} samples)")
    print(f"Test set saved to {test_file} ({len(test_df)} samples)")
    
    return train_file, test_file

def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Prepare dataset for feel-it training")
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--text-col", required=True, help="Name of text column")
    parser.add_argument("--label-col", required=True, help="Name of label column")
    parser.add_argument("--task", choices=["sentiment", "emotion"], required=True, 
                       help="Task type (sentiment or emotion)")
    parser.add_argument("--output", default=None, help="Output CSV file")
    parser.add_argument("--split", action="store_true", help="Split into train/test sets")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size fraction")
    
    args = parser.parse_args()
    
    # Set default output filename
    if args.output is None:
        args.output = f"{args.task}_dataset.csv"
    
    # Prepare dataset
    if args.task == "sentiment":
        df = prepare_sentiment_dataset(args.input, args.text_col, args.label_col, args.output)
    else:
        df = prepare_emotion_dataset(args.input, args.text_col, args.label_col, args.output)
    
    # Split if requested
    if args.split:
        split_dataset(args.output, test_size=args.test_size)

if __name__ == "__main__":
    main() 