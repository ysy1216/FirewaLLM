import torch
from model import BertTextClassifier
from transformers import BertTokenizer, BertConfig
#Y mean sensitive, N mean insensitive
labels = ["Y", "N"]
bert_config = BertConfig.from_pretrained('bert-base-chinese')
# definition model
model = BertTextClassifier(bert_config, len(labels))
# Loading a trained model
model.load_state_dict(torch.load('models/best_model.pkl', map_location=torch.device('cpu')))
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

print('classification: ')
while True:
    text = input('Input: ')
    if not text:
        print('Please enter valid text content！')
        continue

    token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    token_type_ids = token['token_type_ids']

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    model.to('cpu')  # Switch the model to the CPU
    predicted = model(
        input_ids,
        attention_mask,
        token_type_ids,
    )
    pred_label = torch.argmax(predicted, dim=1)

    print('Label:', labels[pred_label])