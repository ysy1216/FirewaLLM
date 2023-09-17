# coding: utf-8
import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertTextClassifier, BertLstmClassifier
from dataset import InfoDataset
from tqdm import tqdm
from sklearn import metrics
import pandas as pd 
import numpy as np

def main():
    # Parameter settings
    batch_size = 64

    epochs = 50
    learning_rate = 1e-4
    # Obtain dataset

    train_dataset = InfoDataset("data/info/train.txt")
    valid_dataset = InfoDataset("data/info/test.txt")
    # Generate Batch
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # Read BERT's configuration file
    bert_config = BertConfig.from_pretrained('bert-base-chinese')
    num_labels = len(train_dataset.labels)
    # initial model     
    model = BertLstmClassifier(bert_config, num_labels).to(device)  #bert_model
    # optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # loss function
    criterion = nn.CrossEntropyLoss()
    best_f1 = 0
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    for epoch in range(1, epochs + 1):
        total_loss = 0  # Total loss for the epoch
        total_correct = 0  # Total correct predictions for the epoch
        total_samples = 0  # Total processed samples for the epoch

        model.train()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_bar = tqdm(train_dataloader, ncols=100, desc='Epoch {} train'.format(epoch))

        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # Move input tensors to the appropriate device (CPU or GPU)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            label_id = label_id.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(input_ids, token_type_ids, attention_mask)

            # Compute the loss
            loss = criterion(output, label_id)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate the number of correct predictions and the total number of samples
            predicted_labels = torch.argmax(output, dim=1)
            total_correct += torch.sum(predicted_labels == label_id).item()
            total_samples += input_ids.size(0)  # input_ids.size(0) is the batch size

            # Update the progress bar description
            train_bar.set_postfix(loss=loss.item())

        average_loss = total_loss / len(train_dataloader)
        average_accuracy = total_correct / total_samples

        print('Train ACC: {:.4f}\tLoss: {:.4f}'.format(average_accuracy, average_loss))

        # test and verify
        model.eval()
        total_loss = 0
        pred_labels = []
        true_labels = []

        with torch.no_grad():
            valid_bar = tqdm(valid_dataloader, ncols=100, desc='Epoch {} valid'.format(epoch))
            for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                label_id = label_id.to(device)

                output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

                loss = criterion(output, label_id)
                total_loss += loss.item()

                pred_label = torch.argmax(output, dim=1)
                acc = torch.sum(pred_label == label_id).item() / len(pred_label)
                valid_bar.set_postfix(loss=loss.item(), acc=acc)

                pred_labels.extend(pred_label.cpu().numpy().tolist())
                true_labels.extend(label_id.cpu().numpy().tolist())

        average_loss = total_loss / len(valid_dataloader)
        print('Validation Loss: {:.4f}'.format(average_loss))

        # Classification report
        report = metrics.classification_report(true_labels, pred_labels, labels=valid_dataset.labels_id,
                                               target_names=valid_dataset.labels)
        print('* Classification Report:')
        print(report)

        f1 = metrics.f1_score(true_labels, pred_labels, labels=valid_dataset.labels_id, average='micro')

        if not os.path.exists('models'):
            os.makedirs('models')

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'models/sen_model.pkl')
#Check if the dataset meets the specifications
def test_1():
        paths = "data/info/train.txt"
        paths1 = "data/info/test.txt"
        with open(paths, encoding="utf8") as f:
            lines = f.readlines()
            for line in tqdm(lines, ncols=100):
                #print(line)
                #print(line.strip().split("_"))
                print(line.strip().split('_'))
                text, label = line.strip().split('_')
                print(label)
if __name__ == '__main__':
    # test_1() #split data
    main()