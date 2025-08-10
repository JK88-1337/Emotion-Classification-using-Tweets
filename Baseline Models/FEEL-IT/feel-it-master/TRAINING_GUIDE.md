# Feel-It Custom Training Guide

This guide explains how to train and test custom sentiment and emotion classification models using the feel-it library with your own datasets.

## Overview

The feel-it library now supports fine-tuning pre-trained models on your own datasets. This allows you to:

1. **Adapt models to your domain**: Fine-tune on domain-specific text
2. **Improve performance**: Train on larger, more relevant datasets
3. **Customize labels**: Adapt to your specific labeling scheme
4. **Test on your data**: Evaluate performance on your own test sets

## Requirements

Install the additional training dependencies:

```bash
pip install -r requirements_training.txt
```

## Dataset Format

Your dataset should be in CSV format with the following structure:

### For Sentiment Classification:
```csv
text,sentiment
"Sono molto felice oggi!","positive"
"Questo film è terribile.","negative"
"La pizza era deliziosa!","positive"
```

### For Emotion Classification:
```csv
text,emotion
"Sono furioso per quello che è successo!","anger"
"Ho paura di quello che potrebbe accadere.","fear"
"Sono così felice di vederti!","joy"
"Mi sento triste e solo oggi.","sadness"
```

## Quick Start

### 1. Prepare Your Dataset

Use the provided utility script to prepare your dataset:

```bash
# For sentiment classification
python examples/prepare_dataset.py \
    --input your_data.csv \
    --text-col "text_column_name" \
    --label-col "label_column_name" \
    --task sentiment \
    --split

# For emotion classification
python examples/prepare_dataset.py \
    --input your_data.csv \
    --text-col "text_column_name" \
    --label-col "label_column_name" \
    --task emotion \
    --split
```

### 2. Train Your Model

```python
from feel_it.trainer import SentimentTrainer, EmotionTrainer
import pandas as pd

# Load your dataset
df = pd.read_csv("sentiment_dataset.csv")
texts = df['text'].tolist()
labels = df['sentiment'].tolist()

# Train sentiment model
trainer = SentimentTrainer()
train_dataset, val_dataset = trainer.prepare_data(texts, labels)
trainer.train(train_dataset, val_dataset, output_dir="./my_sentiment_model")

# Train emotion model
df = pd.read_csv("emotion_dataset.csv")
texts = df['text'].tolist()
labels = df['emotion'].tolist()

trainer = EmotionTrainer()
train_dataset, val_dataset = trainer.prepare_data(texts, labels)
trainer.train(train_dataset, val_dataset, output_dir="./my_emotion_model")
```

### 3. Use Your Custom Model

```python
from feel_it.custom_classifiers import CustomSentimentClassifier, CustomEmotionClassifier

# Load your custom models
sentiment_classifier = CustomSentimentClassifier("./my_sentiment_model")
emotion_classifier = CustomEmotionClassifier("./my_emotion_model")

# Make predictions
texts = ["Sono molto contento!", "Questa esperienza è stata terribile."]

sentiment_predictions = sentiment_classifier.predict(texts)
emotion_predictions = emotion_classifier.predict(texts)

print("Sentiment:", sentiment_predictions)
print("Emotion:", emotion_predictions)
```

## Complete Example

Run the complete example script:

```bash
python examples/train_custom_model.py
```

This script demonstrates the full workflow including:
- Dataset preparation
- Model training
- Performance evaluation
- Comparison with original models

## Advanced Usage

### Custom Training Parameters

```python
# Customize training parameters
trainer.train(
    train_dataset, 
    val_dataset, 
    output_dir="./custom_model",
    num_epochs=5,           # Number of training epochs
    batch_size=8,           # Batch size (reduce if you run out of memory)
    learning_rate=1e-5      # Learning rate
)
```

### Using Different Base Models

```python
# Use a different pre-trained model
trainer = SentimentTrainer(model_name="dbmdz/bert-base-italian-xxl-cased")
```

### Custom Data Loading

```python
# Load data from different sources
def load_my_data():
    # Your custom data loading logic here
    texts = ["text1", "text2", "text3"]
    labels = ["positive", "negative", "positive"]
    return texts, labels

texts, labels = load_my_data()
train_dataset, val_dataset = trainer.prepare_data(texts, labels)
```

## Dataset Requirements

### Minimum Requirements
- **Text**: Italian text (the models are optimized for Italian)
- **Labels**: 
  - Sentiment: `positive` or `negative`
  - Emotion: `anger`, `fear`, `joy`, or `sadness`
- **Size**: At least 100 samples per class for reasonable performance

### Recommended
- **Size**: 1000+ samples per class
- **Balance**: Relatively balanced class distribution
- **Quality**: Clean, well-labeled text
- **Domain**: Text similar to your target domain

## Performance Tips

### 1. Data Quality
- Clean your text data (remove duplicates, fix typos)
- Ensure consistent labeling
- Balance your dataset if possible

### 2. Training Parameters
- Start with small learning rates (1e-5 to 5e-5)
- Use early stopping to prevent overfitting
- Monitor validation loss during training

### 3. Hardware
- Use GPU if available (significantly faster training)
- Reduce batch size if you run out of memory
- Consider using mixed precision training for large models

### 4. Model Selection
- Start with the default models for good baseline performance
- Experiment with different base models for your specific domain
- Consider ensemble methods for better performance

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Use mixed precision training

2. **Poor Performance**
   - Check data quality and labeling
   - Increase dataset size
   - Try different learning rates
   - Use data augmentation

3. **Overfitting**
   - Reduce model complexity
   - Use early stopping
   - Increase regularization
   - Get more training data

### Getting Help

If you encounter issues:
1. Check the error messages carefully
2. Verify your dataset format
3. Try with the sample dataset first
4. Check the transformers library documentation

## Evaluation

### Metrics
The training automatically computes:
- Accuracy
- Precision, Recall, F1-score per class
- Overall classification report

### Custom Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# Load your test data
test_texts = ["test text 1", "test text 2"]
test_labels = ["positive", "negative"]

# Get predictions
predictions = custom_classifier.predict(test_texts)

# Evaluate
print(classification_report(test_labels, predictions))
print(confusion_matrix(test_labels, predictions))
```

## Model Deployment

### Save and Load Models
```python
# Models are automatically saved during training
# Load them later
classifier = CustomSentimentClassifier("./my_sentiment_model")
```

### Production Use
- Models can be loaded and used in production environments
- Consider model size and inference speed for your use case
- Monitor model performance over time

## License and Attribution

When using fine-tuned models:
- Respect the original model licenses
- Attribute the base models appropriately
- Follow the transformers library license terms

## Contributing

To contribute to the training functionality:
1. Fork the repository
2. Add your improvements
3. Include tests for new features
4. Update documentation
5. Submit a pull request

## Support

For questions and support:
- Check the main feel-it documentation
- Review the transformers library documentation
- Open an issue on the GitHub repository 