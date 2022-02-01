import logging, os, sys, json, torch

import numpy as np
import sklearn
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments
from pytorch_lightning.callbacks import EarlyStopping
import sklearn.metrics as metrics
from pytorch_lightning.callbacks import ModelCheckpoint

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TransformerModel(pl.LightningModule):
    def __init__(self, model_name="dumitrescustefan/bert-base-romanian-cased-v1", lr=2e-05, model_max_length=512, labels=[], regression_target=False):
        super().__init__()
        print("Loading AutoModel [{}] ...".format(model_name))
        if regression_target is True:
            print("\t Optimizing MSE loss ...")
        else:
            print("\t Optimizing BCE loss ...")
        self.model_name = model_name
        self.labels = sorted(labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(model_name, config=self.config)
        
        hidden_size = self.get_hidden_size()
        print(f"\tDetected hidden size is {hidden_size}")
        
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(hidden_size, 7)
        self.regression_target = regression_target

        self.multilabel_loss = nn.BCEWithLogitsLoss()
        self.regression_loss = nn.MSELoss() 
        
        self.lr = lr
        self.model_max_length = model_max_length

        self.train_y, self.train_y_hat, self.train_yr_hat, self.train_loss = [], [], [], []
        self.valid_y, self.valid_y_hat, self.valid_yr_hat, self.valid_loss = [], [], [], []
        self.test_y, self.test_y_hat, self.test_yr_hat, self.test_loss = [], [], [], []

        # add pad token
        self.validate_pad_token()
    
    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")

    
    def get_hidden_size(self):    
        inputs = self.tokenizer("text", return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.size(-1)

    
    def forward(self, texts):
        encoded_texts = self.model(input_ids=texts["input_ids"].to(self.device), attention_mask=texts["attention_mask"].to(self.device), return_dict=True)
        encoded_texts = encoded_texts.last_hidden_state  # [batch_size, seq_len, hidden_size]
        encoded_texts = torch.mean(encoded_texts, dim=1)  # [batch_size, hidden_size]
        encoded_texts = self.dropout(encoded_texts)
        predicted_labels = self.linear(encoded_texts)

        return predicted_labels

    def training_step(self, batch, batch_idx):
        texts, labels, labels_regression = batch

        predicted_labels = self(texts)

        binary_predicted_labels = torch.zeros_like(predicted_labels)
        binary_predicted_labels[torch.sigmoid(predicted_labels) >= 0.5] = 1.

        if self.regression_target:
            loss = self.regression_loss(torch.sigmoid(predicted_labels), labels_regression)
        else:
            loss = self.multilabel_loss(predicted_labels, labels)

        self.train_y_hat.extend(binary_predicted_labels.detach().cpu().numpy())
        self.train_yr_hat.extend(predicted_labels.detach().cpu().numpy())
        self.train_y.extend(labels.detach().cpu().numpy())
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        for label_index, label in enumerate(self.labels):
            y, y_hat = [], []
            for i in range(len(self.train_y)):  # one example at a time, grandpa style
                y.append(self.train_y[i][label_index])
                y_hat.append(self.train_y_hat[i][label_index])
            self.log(f"train/{label}/accuracy_score", sklearn.metrics.accuracy_score(y_hat, y), prog_bar=False)
            self.log(f"train/{label}/recall_score", sklearn.metrics.recall_score(y_hat, y), prog_bar=False)
            self.log(f"train/{label}/precision_score", sklearn.metrics.precision_score(y_hat, y), prog_bar=False)
            self.log(f"train/{label}/f1_score", sklearn.metrics.f1_score(y_hat, y), prog_bar=False)

        self.log("train/hamming_loss", sklearn.metrics.hamming_loss(self.train_y, self.train_y_hat), prog_bar=True)
        self.log("train/accuracy_score", sklearn.metrics.accuracy_score(self.train_y, self.train_y_hat), prog_bar=False)
        self.log("train/f1_score", sklearn.metrics.f1_score(self.train_y, self.train_y_hat, average='micro'), prog_bar=False)
        self.log("train/mse", sklearn.metrics.mean_squared_error(self.train_y, self.train_yr_hat), prog_bar=False)
        self.log("train/avg_loss", sum(self.train_loss) / len(self.train_loss), prog_bar=False)

        self.train_y, self.train_y_hat, self.train_yr_hat, self.train_loss = [], [], [], []

    def validation_step(self, batch, batch_idx):
        texts, labels, labels_regression = batch

        predicted_labels = self(texts)

        binary_predicted_labels = torch.zeros_like(predicted_labels)
        binary_predicted_labels[torch.sigmoid(predicted_labels) >= 0.5] = 1.

        if self.regression_target:
            loss = self.regression_loss(torch.sigmoid(predicted_labels), labels_regression)
        else:
            loss = self.multilabel_loss(predicted_labels, labels)

        self.valid_y_hat.extend(binary_predicted_labels.detach().cpu().numpy())
        self.valid_yr_hat.extend(predicted_labels.detach().cpu().numpy())
        self.valid_y.extend(labels.detach().cpu().numpy())
        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        for label_index, label in enumerate(self.labels):
            y, y_hat = [], []
            for i in range(len(self.valid_y)): # one example at a time, grandpa style
                y.append(self.valid_y[i][label_index])
                y_hat.append(self.valid_y_hat[i][label_index])
            self.log(f"valid/{label}/accuracy_score", sklearn.metrics.accuracy_score(y_hat, y), prog_bar=False)
            self.log(f"valid/{label}/recall_score", sklearn.metrics.recall_score(y_hat, y), prog_bar=False)
            self.log(f"valid/{label}/precision_score", sklearn.metrics.precision_score(y_hat, y), prog_bar=False)
            self.log(f"valid/{label}/f1_score", sklearn.metrics.f1_score(y_hat, y), prog_bar=False)

        self.log("valid/hamming_loss", sklearn.metrics.hamming_loss(self.valid_y, self.valid_y_hat), prog_bar=True)
        self.log("valid/accuracy_score", sklearn.metrics.accuracy_score(self.valid_y, self.valid_y_hat), prog_bar=True)
        self.log("valid/f1_score", sklearn.metrics.f1_score(self.valid_y, self.valid_y_hat, average='micro'), prog_bar=False)
        self.log("valid/mse", sklearn.metrics.mean_squared_error(self.valid_y, self.valid_yr_hat), prog_bar=False)
        self.log("valid/avg_loss", sum(self.valid_loss) / len(self.valid_loss), prog_bar=False)

        self.valid_y, self.valid_y_hat, self.valid_yr_hat, self.valid_loss = [], [], [], []


    def test_step(self, batch, batch_idx):
        texts, labels, labels_regression = batch

        predicted_labels = self(texts)

        binary_predicted_labels = torch.zeros_like(predicted_labels)
        binary_predicted_labels[torch.sigmoid(predicted_labels) >= 0.5] = 1.

        if self.regression_target:
            loss = self.regression_loss(torch.sigmoid(predicted_labels), labels_regression)
        else:
            loss = self.multilabel_loss(predicted_labels, labels)

        self.test_y_hat.extend(binary_predicted_labels.detach().cpu().numpy())
        self.test_yr_hat.extend(predicted_labels.detach().cpu().numpy())
        self.test_y.extend(labels.detach().cpu().numpy())
        self.test_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def test_epoch_end(self, outputs):
        for label_index, label in enumerate(self.labels):
            y, y_hat = [], []
            for i in range(len(self.test_y)):  # one example at a time, grandpa style
                y.append(self.test_y[i][label_index])
                y_hat.append(self.test_y_hat[i][label_index])
            self.log(f"test/{label}/accuracy_score", sklearn.metrics.accuracy_score(y_hat, y), prog_bar=False)
            self.log(f"test/{label}/recall_score", sklearn.metrics.recall_score(y_hat, y), prog_bar=False)
            self.log(f"test/{label}/precision_score", sklearn.metrics.precision_score(y_hat, y), prog_bar=False)
            self.log(f"test/{label}/f1_score", sklearn.metrics.f1_score(y_hat, y), prog_bar=False)

        self.log("test/hamming_loss", sklearn.metrics.hamming_loss(self.test_y, self.test_y_hat), prog_bar=True)
        self.log("test/accuracy_score", sklearn.metrics.accuracy_score(self.test_y, self.test_y_hat), prog_bar=True)
        self.log("test/f1_score", sklearn.metrics.f1_score(self.test_y, self.test_y_hat, average='micro'), prog_bar=False)
        self.log("test/mse", sklearn.metrics.mean_squared_error(self.test_y, self.test_yr_hat), prog_bar=False)
        self.log("test/avg_loss", sum(self.test_loss) / len(self.test_loss), prog_bar=False)

        self.test_y, self.test_y_hat, self.test_yr_hat, self.test_loss = [], [], [], []


    def configure_optimizers(self):
            return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)


class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path: str):
        self.file_path = file_path
        self.instances = []
        self.tokenizer = tokenizer
        print(f"Reading file: {file_path}")

        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)

        # compute word fertility rate
        tokens, words = 0, 0
        for example in data:
            tweet = example["text"].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").strip()
            t = self.tokenizer(tweet)
            tokens += len(t['input_ids'])
            words += len(tweet.split())

        print(f"\t Word Fertility Rate = {tokens/words}, from {tokens} tokens and {words} words")

        for example in data:
            tweet = example["text"].replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș").strip()
            labels = example["agreed_labels"]
            normalized_labels = example['procentual_labels']

            instance = {
                "text": tweet,
                "labels": labels,
                "labels_regression": normalized_labels
            }
            self.instances.append(instance)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]

    def custom_collate(self, batch):
        """ texts is a dict like:
        'input_ids': tensor([[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                             [101, 1262, 1330, 5650, 102, 0, 0, 0, 0],
                             [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 0]]),
        'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1],
        """
        texts = []
        labels = []
        labels_regression = []

        for instance in batch:
            # print(instance["sentence1"])
            texts.append(instance["text"])
            labels.append(instance["labels"])
            labels_regression.append(instance["labels_regression"])

        texts = self.tokenizer(texts, padding=
        True, max_length=512, truncation=True, return_tensors="pt")
        labels = torch.tensor(labels, dtype=torch.float)
        labels_regression = torch.tensor(labels_regression, dtype=torch.float)

        return texts, labels, labels_regression


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--model_name', type=str,
                        default="dumitrescustefan/bert-base-romanian-cased-v1")  # xlm-roberta-base
    parser.add_argument('--lr', type=float, default=1e-05)
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--experiment_iterations', type=int, default=1)
    parser.add_argument('--regression', action='store_true')

    args = parser.parse_args()
    #args.regression = True
    print("Batch size is {}, accumulate grad batches is {}, final batch_size is {}\n".format(args.batch_size,
                                                                                             args.accumulate_grad_batches,
                                                                                             args.batch_size * args.accumulate_grad_batches))

    model = TransformerModel(model_name=args.model_name, lr=args.lr,
                             model_max_length=args.model_max_length,
                             labels=[],
                             regression_target=args.regression
                             )
        

    print("Loading data...")
    train_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../data/train.json")
    val_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../data/valid.json")
    test_dataset = MyDataset(tokenizer=model.tokenizer, file_path="../data/test.json")

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True,
                                  collate_fn=train_dataset.custom_collate, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                collate_fn=val_dataset.custom_collate, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False,
                                 collate_fn=test_dataset.custom_collate, pin_memory=True)

    print("Train dataset has {} instances.".format(len(train_dataset)))
    print("Valid dataset has {} instances.".format(len(val_dataset)))
    print("Test dataset has {} instances.\n".format(len(test_dataset)))

    itt = 0

    v_hamming_loss, v_accuracy_score, v_mse, t_hamming_loss, t_accuracy_score, t_mse = [], [], [], [], [], []

    while itt < args.experiment_iterations:
        print("Running experiment {}/{}".format(itt + 1, args.experiment_iterations))

        model = TransformerModel(model_name=args.model_name,
                                 lr=args.lr,
                                 model_max_length=args.model_max_length,
                                 labels=list(range(7)),
                                 regression_target=args.regression
                                 )

        early_stop = EarlyStopping(
            monitor='valid/hamming_loss' if not args.regression else 'valid/avg_loss',
            patience=5,
            verbose=True,
            mode='min'
        )

        trainer = pl.Trainer(
            gpus=args.gpus,
            callbacks=[early_stop],
            #limit_train_batches=5,
            #limit_val_batches=2,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=1.0,
            enable_checkpointing=False
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        result_valid = trainer.test(model, val_dataloader)
        result_test = trainer.test(model, test_dataloader)

        with open("results_{}_of_{}_{}_regression_{}.json".format(itt + 1, args.experiment_iterations, args.model_name.replace("/", "_"), args.regression), "w") as f:
            json.dump(result_test[0], f, indent=4, sort_keys=True)

        v_hamming_loss.append(result_valid[0]['test/hamming_loss'])
        v_accuracy_score.append(result_valid[0]['test/accuracy_score'])
        v_mse.append(result_valid[0]['test/mse'])
        t_hamming_loss.append(result_test[0]['test/hamming_loss'])
        t_accuracy_score.append(result_test[0]['test/accuracy_score'])
        t_mse.append(result_test[0]['test/mse'])

        itt += 1

    print("Done, writing results...")
    result = {}
    result["valid_hamming_loss"] = sum(v_hamming_loss) / args.experiment_iterations
    result["valid_accuracy_score"] = sum(v_accuracy_score) / args.experiment_iterations
    result["valid_mse"] = sum(v_mse) / args.experiment_iterations
    result["test_hamming_loss"] = sum(t_hamming_loss) / args.experiment_iterations
    result["test_accuracy_score"] = sum(t_accuracy_score) / args.experiment_iterations
    result["test_mse"] = sum(t_mse) / args.experiment_iterations

    with open("results_{}_regression_{}.json".format(args.model_name.replace("/", "_"), args.regression), "w") as f:
        json.dump(result, f, indent=4, sort_keys=True)

    print("REDv2 averaged test results:")
    print("_"*80+"\n")
    from pprint import pprint
    pprint(result)