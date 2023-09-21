import torch
import torch.nn as nn
from transformers import BertModel

# Bert
class BertTextClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        # Define BERT model
        self.bert = BertModel(config=bert_config)
        # Define Classifier
        self.classifier = nn.Linear(bert_config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # BERT's output
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Pooled output at [CLS] position
        pooled = bert_output[1]
        # classification
        logits = self.classifier(pooled)
        # Return the result after softmax
        return torch.softmax(logits, dim=1)


# Bert+BiLSTM
class BertLstmClassifier(nn.Module):
    def __init__(self, bert_config, num_labels):
        super().__init__()
        self.bert = BertModel(config=bert_config)
        self.lstm = nn.LSTM(input_size=bert_config.hidden_size, hidden_size=bert_config.hidden_size, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(bert_config.hidden_size * 2, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        out, _ = self.lstm(last_hidden_state)
        logits = self.classifier(out[:, -1, :])  # Take the output at the last moment
        return self.softmax(logits)