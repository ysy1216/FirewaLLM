import torch.nn as nn
from transformers import BertModel, BertConfig
from transformers import BertForSequenceClassification
# Bert
class BertTextClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        """
        Args:
            bert_model_name (str): BERTThe name or path of the model
            num_labels (int): Number of categories classified
        """
        super(BertTextClassifier, self).__init__()
        # Define BERT model
        bert_config = BertConfig.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name, config=bert_config)
        # Define Classifier
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT's output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = bert_output[1]
        # classification
        logits = self.classifier(pooled)
        return logits
# Bert+BiLSTM
class BertLstmClassifier(nn.Module):
    def __init__(self, bert_model_name, num_labels):
        super(BertLstmClassifier, self).__init__()
        # Using BertForSequenceClassification in the Hugging Face Transformers library
        self.bert = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_labels)
        # 双向LSTM层
        self.lstm = nn.LSTM(input_size=BertConfig.hidden_size, hidden_size=BertConfig.hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Using BERT for feature extraction
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Extract BERT's output
        bert_output = outputs.logits
        # Using bidirectional LSTM for processing
        lstm_output, _ = self.lstm(bert_output)
        # Take the output at the last moment
        logits = lstm_output[:, -1, :]
        return logits
