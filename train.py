import os
import pandas as pd
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Bert import BertConfig, Bert
from transformers import BertTokenizer
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
lr = 1e-5
epochs = 30
config = BertConfig()
train_size = 0.8
#tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config.num_labels = 5
config.vocab_size = 30521
label_ls = ["False", "True"]


def encode_text(text):
    encoded_dict = tokenizer.encode_plus(
                        text,                      # Input text
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 128,           # Pad & truncate all sentences
                        truncation = True,
                        padding = 'max_length',         # Pad all sentences
                        return_attention_mask = True,   # Construct attention masks
                        return_tensors = "pt",     # Return pytorch tensors
                   )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']


class CustomData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): # item
        item = self.data[idx]
        utterance = item['utterance']
        context = item['context']
        utterance_and_context = ' '.join([sentence for sentence in context] + [utterance]) # Combining the utterance and its context into one string.
        sarcasm = int(item['sarcasm'])
        #input_ids, attention_mask = encode_text(utterance_and_context)
        return utterance_and_context, sarcasm


    @staticmethod
    def call_fc(batch):

        utterances = []
        labels = []

        for utterance, label in batch:
            utterances.append(utterance)
            labels.append(label)

        inputs = tokenizer.batch_encode_plus(utterances, padding=True, truncation="only_first", max_length=128, return_tensors="pt")
        labels = np.array(labels, dtype="int64")

        labels = torch.from_numpy(labels)

        return inputs, labels



def train():
    with open('data/train_mixed.json') as file:
        train_data = json.load(file)

    with open('data/val_mixed.json') as file:
        val_data = json.load(file)

    with open('data/test_mixed.json') as file:
        test_data = json.load(file)

    # Convert the data to a list of dictionaries
    train_data = list(train_data.values())
    val_data = list(val_data.values())
    test_data = list(test_data.values())

    train_data = CustomData(train_data)
    train_data = DataLoader(train_data, shuffle=True, batch_size=len(train_data), collate_fn=train_data.call_fc)

    val_data = CustomData(val_data)
    val_data = DataLoader(val_data, shuffle=True, batch_size=len(val_data), collate_fn=val_data.call_fc)
    
    test_data = CustomData(test_data)
    test_data = DataLoader(test_data, shuffle=True, batch_size=len(test_data), collate_fn=test_data.call_fc)

    print("Data Loaded")

    model = Bert(config)
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()

    loss_old = 100

    train_result = []
    val_result = []
    test_result = []

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_data)
        loss_all = 0
        acc_all = 0
        print("Epoch", epoch)
        for step, (x, y) in enumerate(pbar): # Iterate over batches
            print("Batch")
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            loss = loss_fc(out, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_all += loss.item()
            #loss_time = loss_all / (step + 1)

            acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

            acc_all += acc
            #acc_time = acc_all / (step + 1)

            #s = "train => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(epoch, step, loss, loss_time, acc, acc_time)
            s = "train => epoch:{} - loss:{:.4f} - acc:{:.4f}".format(epoch, loss, acc)
            #pbar.set_description(s)

            train_result.append(s+"\n")
        # end of training loop

        # Validation loop
        val_loss_all = 0
        val_acc_all = 0
        #val_steps = 0
        for x, y in val_data:
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            loss = loss_fc(out, y)

            val_loss_all += loss.item()
            val_acc_all += torch.mean((y == torch.argmax(out, dim=-1)).float())
            #val_steps += 1
        # end val loop

        val_loss = val_loss_all # / val_steps
        val_acc = val_acc_all # / val_steps

        val_result.append(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}\n")

        with torch.no_grad():
            #pbar = tqdm(test_data)
            loss_all = 0
            acc_all = 0
            #for step, (x, y) in enumerate(pbar):
            for x, y in test_data:
                x = {k: v.to(device) for k, v in x.items()}
                y = y.to(device)
                out = model(**x)

                loss = loss_fc(out, y)

                loss_all += loss.item()
                test_loss_time = loss_all #/ (step + 1)

                acc = torch.mean((y == torch.argmax(out, dim=-1)).float())

                acc_all += acc
                acc_time = acc_all #/ (step + 1)

                #s = "test => epoch:{} - step:{} - loss:{:.4f} - loss_time:{:.4f} - acc:{:.4f} - acc_time:{:.4f}".format(epoch, step, loss, test_loss_time, acc, acc_time)
                s = "test => epoch:{} - loss:{:.4f} - acc:{:.4f}".format(epoch, loss, acc)
                #pbar.set_description(s)

                test_result.append(s+"\n")  # Append test results to test_result list
            # end of test loop

        with open("train_result.txt", "w") as f:
            f.writelines(train_result)

        with open("val_result.txt", "w") as f:
            f.writelines(val_result)

        with open("test_result.txt", "w") as f:
            f.writelines(test_result)  # Write test_result to file

        if loss_old > test_loss_time:
            loss_old = test_loss_time
            torch.save(model.state_dict(), "model.pkl")

    ys = []
    prs = []

    model.load_state_dict(torch.load("model.pkl"))
    model.eval()

    with torch.no_grad():
        #pbar = tqdm(test_data)
        #for step, (x, y) in enumerate(pbar):
        for x, y in test_data:
            x = {k:v.to(device) for k, v in x.items()}
            y = y.to(device)
            out = model(**x)

            p = torch.argmax(out, dim=-1).cpu().numpy()
            y = y.cpu().numpy()

            ys.append(y)
            prs.append(p)
        # end loop

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(prs, axis=0)

    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"precision：{precision*100}%")
    print(f"recall：{recall*100}%")
    print(f"f1：{f1*100}%")


    cm = confusion_matrix(y_true, y_pred)
    true_labels = np.unique(y_true)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(true_labels)), true_labels)
    plt.yticks(np.arange(len(true_labels)), true_labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion matrix')

    for i in range(len(true_labels)):
        for j in range(len(true_labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='red')

    plt.show()

if __name__ == '__main__':
    train()

