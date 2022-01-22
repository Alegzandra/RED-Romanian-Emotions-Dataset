import pandas as pd
import nltk
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch_dataset_creator import RED
from torch.utils.data import Dataset, DataLoader
import re
from BERT_EmotionClassifier import EmotionsClassifier
from torch import nn, optim
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


PRE_TRAINED_MODEL_NAME = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

EPOCHS = 5
#RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_LEN = 100

df_train = pd.read_csv(r"\RED data\train.csv")
df_val = pd.read_csv(r"\RED data\val.csv")
df_test = pd.read_csv(r"\RED data\test.csv")
print(df_train.shape, df_test.shape, df_val.shape)

#df[["content", "emotion"]].groupby("emotion").count().plot(kind="bar")
#plt.show()

class_names=["Bucurie", "Furie", "Frica", "Tristete", "Neutru"]

#labels must be of type int
label_encoder = LabelEncoder()

df_train.Emotion = label_encoder.fit_transform(df_train.Emotion)
df_val.Emotion = label_encoder.transform(df_val.Emotion)
df_test.Emotion = label_encoder.transform(df_test.Emotion)

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = RED(
    texts=df.Tweet.to_numpy(),
    emotions=df.Emotion.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=0,

  )


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
  model = model.train()
  losses = []
  correct_predictions = 0
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    emotions = d["emotions"].to(device)
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, emotions)
    correct_predictions += torch.sum(preds == emotions)
    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()
  losses = []
  correct_predictions = 0
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      emotions = d["emotions"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      loss = loss_fn(outputs, emotions)
      correct_predictions += torch.sum(preds == emotions)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

print(train_data_loader)


data = next(iter(train_data_loader))
#print(data['input_ids'].shape)


model = EmotionsClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
model.to(device)


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)
F.softmax(model(input_ids, attention_mask), dim=1)

optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

#loss_fn = nn.NLLLoss().to(device)
loss_fn = nn.CrossEntropyLoss().to(device)
history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    #len(texts_train)
    len(df_train)
  )
  print(f'Training loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
    model,
    val_data_loader,
    loss_fn,
    device,
    #len(texts_val)
    len(df_val)
  )

  print(f'Validation loss {val_loss} accuracy {val_acc}')
  print()
  
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)

  if val_acc > best_accuracy:
    torch.save(model.state_dict(), 'best_model_state_new5.bin')
    best_accuracy = val_acc

plt.plot(history['train_acc'], label='train accuracy')
plt.plot(history['val_acc'], label='validation accuracy')

plt.title('Training history')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 1])
plt.show()
