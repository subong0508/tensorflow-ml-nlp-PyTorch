import os
import json
import numpy as np
import pandas as pd
from konlpy.tag import Okt
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from transformers import BertForMaskedLM

# BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_basic_tokenize=True)
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')

input_sentences = pd.read_csv('./data_in/ChatbotData.csv')['Q']
label_sentences = pd.read_csv('./data_in/ChatbotData.csv')['A']

# preprocessing
def get_max_len(input_sentences=input_sentences, label_sentences=label_sentences):
    answer = 0
    for inp, out in zip(input_sentences, label_sentences):
        len1 = len(tokenizer.tokenize(inp))
        len2 = len(tokenizer.tokenize(out))
        maxlen = len1 + len2
        if maxlen > answer:
            answer = maxlen

    return answer

MAX_LEN = get_max_len() + 3 # [CLS], [SEP], [SEP]


def tokenize(pair: tuple):
    input_sentence, label_sentence = pair
    input_sentence = " [CLS] " + input_sentence + " [SEP] "
    input_text, label_text = tokenizer.tokenize(input_sentence), tokenizer.tokenize(label_sentence)

    for _ in range(len(label_text)):
        input_text.append("[MASK]")

    for _ in range(MAX_LEN - len(input_text)):
        input_text.append("[MASK]")

    input_tokens = tokenizer.convert_tokens_to_ids(input_text)
    input_tensor = torch.tensor([input_tokens])

    label_ids = [-100] * len(tokenizer.tokenize(input_sentence))

    label_ids += tokenizer.convert_tokens_to_ids(label_text)
    label_ids.append(tokenizer.convert_tokens_to_ids(['[SEP]'])[0])

    for _ in range(MAX_LEN - len(label_ids)):
        label_ids.append(-100)
    label_tensor = torch.tensor([label_ids])

    return [input_tensor, label_tensor]

# prepare data
class Data(Dataset):
  def __init__(self):
    self.input_sentences = []
    self.label_sentences = []
    self.prepare()

  def prepare(self):
    for input_sentence, label_sentence in zip(input_sentences, label_sentences):
      inp, out = tokenize((input_sentence, label_sentence))
      self.input_sentences.append(inp)
      self.label_sentences.append(out)

  def __getitem__(self, s):
    return self.input_sentences[s], self.label_sentences[s]

  def __len__(self):
    return len(self.input_sentences)

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
lr = 5e-5

data = Data()
data_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = optim.Adamax(model.parameters(), lr=lr)

# train
def train(model=model, optimizer=optimizer, num_epochs=EPOCHS):
    model.train()
    losses = []

    for epoch in range(1, num_epochs + 1):
        for idx, (inp, out) in enumerate(data_loader):
            inp, out = inp.to(device), out.to(device)
            inp, out = inp.squeeze(), out.squeeze()

            loss = model(inp, masked_lm_labels=out)[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 5 == 1 and idx % 20 == 0:
                print(f"Epoch: {epoch}, Batch: {idx + 1}, Loss: {loss.item()}")

        losses.append(loss.item())
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print("=" * 50)

    return losses


# predict
def infer(filepath):
    model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
    model.load_state_dict(torch.load(filepath))
    model.eval()
    while True:
        print("입력하세요: ")
        question = input()
        if question == 'q': break
        question = " [CLS] " + question + " [SEP] "
        tokenized_text = tokenizer.tokenize(question)
        for _ in range(MAX_LEN - len(tokenized_text)):
            tokenized_text.append("[MASK]")

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        answer = []
        with torch.no_grad():
            predictions = model(tokens_tensor)[0]
            predictions = predictions.squeeze()
            predicted_index = torch.argmax(predictions, dim=-1)
            print(predicted_index.shape)

            for index in predicted_index:
                answer.append(index.item())

            print(answer)

            if 102 in answer:  # [SEP]
                end = answer.index(102)
            else:
                end = len(answer)

            answer = answer[:end]
            print(tokenizer.decode(answer))

if __name__ == "__main__":
    losses = train()

    torch.save(model.state_dict(), "./BERT_finetune.pth")
    print("Model is saved.")

    filepath = input("모델 저장경로를 입력하세요: ")
    infer(filepath)















