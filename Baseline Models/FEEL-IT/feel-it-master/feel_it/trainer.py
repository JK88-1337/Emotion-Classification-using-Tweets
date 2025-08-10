import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import numpy as np
from feel_it.dataset import TextDataset

class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class EmotionTrainer:
    def __init__(self, model_name="MilaNLProc/feel-it-italian-emotion"):
        """
        Trainer for fine-tuning emotion classification models
        """
        self.model_name = model_name
        self.emotion_map = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}
        self.reverse_emotion_map = {v: k for k, v in self.emotion_map.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
    def prepare_data(self, texts, labels, test_size=0.2, random_state=42):
        """
        Prepare data for training
        
        Args:
            texts: List of text strings
            labels: List of labels (should be 'anger', 'fear', 'joy', or 'sadness')
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Convert string labels to integers
        numeric_labels = [self.reverse_emotion_map[label] for label in labels]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, numeric_labels, test_size=test_size, random_state=random_state, stratify=numeric_labels
        )
        
        # Tokenize
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=500)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True, max_length=500)
        
        # Create datasets
        train_dataset = CustomTextDataset(train_encodings, train_labels)
        val_dataset = CustomTextDataset(val_encodings, val_labels)
        
        return train_dataset, val_dataset
    
    def train(self, train_dataset, val_dataset, output_dir="./emotion_model", 
              num_epochs=3, batch_size=16, learning_rate=2e-5):
        """
        Fine-tune the emotion model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for training
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            learning_rate=learning_rate,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer 