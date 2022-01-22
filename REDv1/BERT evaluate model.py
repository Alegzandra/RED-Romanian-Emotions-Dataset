from BERT_EmotionClassifier import EmotionsClassifier
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch_dataset_creator import RED
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import seaborn as sn
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.preprocessing import LabelEncoder
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import itertools
from sklearn.metrics import classification_report

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
RANDOM_SEED = 42
BATCH_SIZE = 8
MAX_LEN = 100

#denumirea emotiilor in ordine alfabetica
class_names=["Bucurie", "Frica", "Furie", "Neutru", "Tristete"]
#class_names=["bucurie", "furie", "frica", "tristete", "neutral"]

PRE_TRAINED_MODEL_NAME = 'dumitrescustefan/bert-base-romanian-cased-v1'
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

#deschid modelul
model = EmotionsClassifier(len(class_names), PRE_TRAINED_MODEL_NAME)
model.load_state_dict(torch.load('best_model_state_new5.bin', map_location=torch.device('cpu')))
model = model.to(device)

#deschid datele de test
#df_train = pd.read_csv("dataset/RED_train.csv")
#df_val = pd.read_csv("dataset/RED_val.csv")
#df_test = pd.read_csv("dataset/RED_test.csv")
df_test = pd.read_csv(r"D:\WFH\untitled\venv\new\test.csv")

label_encoder = LabelEncoder()
#potrivim encoderul pe datele de train?
#df_train.emotion = label_encoder.fit_transform(df_train.emotion)
#df_val.emotion = label_encoder.transform(df_val.emotion)
df_test.Emotion = label_encoder.fit_transform(df_test.Emotion)

#creez data loader
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
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

def get_predictions(model, data_loader):
    model = model.eval()

    question_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["question_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            # probabilities
            probs = F.softmax(outputs, dim=1)

            question_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return question_texts, predictions, prediction_probs, real_values

class Bert_model:
    def __init__(self, model, encoder, tokenizer, maxlen):
        self._model = model
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._device = "cpu"
        self._maxlen = maxlen

    def set_device(self, device):
        self._device = device
        self._model.to(device)

    def _preprocess_text(self, text):
        # eliminate URLs
        result = re.result = re.sub(r"http\S+", "", text)
        # eliminate email addresses
        result = re.sub('\S*@\S*\s?', '', result)
        return result

    def _make_prediction_for_text(self, text):
        text = self._preprocess_text(text)
        encoding = self._tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self._maxlen,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self._device)
        attention_mask = encoding['attention_mask'].to(self._device)
        outputs = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        outputs.to(self._device)
        return outputs.cpu().detach().numpy()

    def detect_emotions_with_labels(self, texts):
        predicted_emotions = []
        for text in texts:
            emotion = self._make_prediction_for_text(text)
            pred = np.argmax(emotion, axis=1)
            prediction = self._encoder.inverse_transform(pred)
            predicted_emotions.append(prediction[0])
        return predicted_emotions

    def detect_emotions_with_proba(self, texts):
        predicted_emotions = []
        for text in texts:
            emotion_proba = self._make_prediction_for_text(text)
            predicted_emotions.append(emotion_proba)
        return np.concatenate(predicted_emotions)

#tweet-urile din test
texts_test = df_test.Tweet.tolist()
#dataframe-ul cu emotiile, labeluite
emotions_test = df_test.Emotion
print("emotions_test", emotions_test)
#creez obiectul model bert
bert_model = Bert_model(model, label_encoder, tokenizer, MAX_LEN)
#creez vectorul de predictiile modelului pe tweet-urile de test
bert_predictions = bert_model.detect_emotions_with_labels(texts_test)
print('bert_predictions', bert_predictions, len(bert_predictions))
y_true = label_encoder.inverse_transform(emotions_test)
print('y_true', y_true, len(y_true))

#compute_metrics(y_true, bert_predictions)
true = []
for elem in y_true:
    true.append(elem)

#matricea de confuzie
cm = confusion_matrix(true, bert_predictions, labels=class_names)

print(true)
print(bert_predictions)

print(cm)

#matricea de confuzie normalizata
cm2 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm)

df_cm = pd.DataFrame(cm2, index = [i for i in class_names], columns = [i for i in class_names])
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.4) # for label size
#add fmt = 'd' f
sn.heatmap(df_cm, annot=True, cmap="Blues") #annot_kws={"size": 16}
plt.show()

acc = accuracy_score(true, bert_predictions)
precision = precision_score(true, bert_predictions, average='macro')
recall = recall_score(true, bert_predictions, average='macro')
f1 = f1_score(true, bert_predictions, average='macro')

print('acc - ', acc, 'prec - ', precision, 'recall - ', recall, 'f-score - ', f1)
print(classification_report(true, bert_predictions, target_names=class_names))
exit(0)
while True:
    review_text = input()

    encoded_review = tokenizer.encode_plus(
        review_text,
        max_length=MAX_LEN,
        add_special_tokens=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)

    prob = F.softmax(output, dim=1)
    top_p, top_class = prob.topk(1, dim=1)

    print(f'Review text: {review_text}')
    # print(f'Intent  : {class_names[prediction]}' )
    print('Intent: ', class_names[top_class])
    print('Probability: ', top_p)

