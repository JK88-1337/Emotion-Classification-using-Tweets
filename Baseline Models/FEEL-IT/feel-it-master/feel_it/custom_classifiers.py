from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from feel_it.dataset import TextDataset

class CustomSentimentClassifier:
    def __init__(self, model_path):
        """
        Custom sentiment classifier that loads a fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.sentiment_map = {0: "negative", 1: "positive"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def predict(self, sentences, batch_size=32):
        """
        Predicts the sentiment for the sentences in input
        @param sentences: sentences to be classified with the sentiment classifier
        @param batch_size: batch size for the network
        @return: List of predicted sentiments
        """
        train_encodings = self.tokenizer(sentences,
                                    truncation=True,
                                    padding=True,
                                    max_length=500)

        train_dataset = TextDataset(train_encodings)

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        collect_outputs = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                collect_outputs.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())

        return [self.sentiment_map[k] for k in collect_outputs]

class CustomEmotionClassifier:
    def __init__(self, model_path):
        """
        Custom emotion classifier that loads a fine-tuned model
        
        Args:
            model_path: Path to the fine-tuned model directory
        """
        self.emotion_map = {0: "anger", 1: "fear", 2: "joy", 3: "sadness"}
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def predict(self, sentences, batch_size=32):
        """
        Predicts the emotion for the sentences in input
        @param sentences: sentences to be classified with the emotion classifier
        @param batch_size: batch size for the network
        @return: List of predicted emotions
        """
        train_encodings = self.tokenizer(sentences,
                                    truncation=True,
                                    padding=True,
                                    max_length=500)

        train_dataset = TextDataset(train_encodings)

        loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        collect_outputs = []

        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                collect_outputs.extend(torch.argmax(outputs["logits"], axis=1).cpu().numpy().tolist())

        return [self.emotion_map[k] for k in collect_outputs] 