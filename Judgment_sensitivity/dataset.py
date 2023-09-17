import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class InfoDataset(Dataset):
    def __init__(self, filename):
        self.labels = ["Y", "N"]
        self.labels_id = list(range(len(self.labels)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.data = []
        self.load_data(filename)

    def load_data(self, filename):
        print('Loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            for line in rf:
                text, label = line.strip().split('_')
                label_id = self.labels.index(label)
                token = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
                input_ids = torch.tensor(token['input_ids'])
                token_type_ids = torch.tensor(token['token_type_ids'])
                attention_mask = torch.tensor(token['attention_mask'])

                self.data.append((input_ids, token_type_ids, attention_mask, label_id))

    def __getitem__(self, index):
        input_ids, token_type_ids, attention_mask, label_id = self.data[index]
        return input_ids, token_type_ids, attention_mask, label_id

    def __len__(self):
        return len(self.data)
