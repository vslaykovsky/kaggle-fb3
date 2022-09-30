# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.007772, "end_time": "2022-09-09T15:49:40.515967", "exception": false, "start_time": "2022-09-09T15:49:40.508195", "status": "completed"} tags=[]
# # Directory settings

# %% [markdown] papermill={"duration": 0.006718, "end_time": "2022-09-09T15:49:40.557772", "exception": false, "start_time": "2022-09-09T15:49:40.551054", "status": "completed"} tags=[]
# # CFG

# %%
import sys

#Benchmark column scores:
['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

#baseline
[
    0.48251612898722873,
    0.44163526218410554,
    0.40993880075686456,
    0.4535148230982543,
    0.4667161054325501,
    0.439366242020053
]


# %%
try:
    import colab
    # only install in colab
    # !pip install transformers tokenizers sentencepiece optuna wandb
except:
    pass

# %%
"""
{'epochs': 3, 'pct_warmup_steps': 0.08531545137950809, 'decoder_lr': 0.005477780636724191, 'encoder_lr': 5e-06, 'skip_connection': False, 'per_column_lstm': True, 'scheduler': 'cosine', 'gradient_accumulation_steps': 8, 'emb_type': 'mean'}.
Best is trial 3 with value: 0.450561181167652.

"""


class CFG:
    wandb = True
    debug = False
    train = False
    optuna = True
    optuna_trials = 30
    cross_validation = False
    unscrew_names = False

    prediction_type = 'regression'  # ['regression', 'classification']
    scheduler = 'cosine'  # ['linear', 'cosine', 'onecycle']
    cosine_num_cycles = 0.5
    batch_scheduler = True
    pct_warmup_steps = 0.1
    encoder_lr = 5e-6  # [5e-6, 8e-6, 9e-6, 1e-5]
    decoder_lr = 5e-3
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    head_weight_decay = 0.01
    max_grad = 1.
    lstm_dropout = 0.1  # [0,0.15,0.3]
    skip_connection = False

    apex = True
    num_workers = 12
    model = "microsoft/deberta-v3-base"
    epochs = 3
    batch_size = 2
    gradient_checkpointing = False
    gradient_accumulation_steps = 8  # [8, 16, 32]
    max_len = 512
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    seed = 42
    n_fold = 4
    # trn_fold=[1]
    trn_fold = [0, 1, 2, 3]
    emb_type = 'mean'  #'lstm'  # ['lstm', 'mean']
    frozen_backbone = False


if CFG.debug:
    CFG.epochs = 4
    CFG.trn_fold = [0]

# %%
import copy
import gc
import warnings

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
except:
    # !pip install iterative-stratification == 0.1.7
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, default_collate
import math

import matplotlib.pyplot as plt

try:
    import transformers
except:
    # !pip uninstall -y transformers
    # !pip uninstall -y tokenizers
    # !python -m pip install --no-index --find-links=../input/fb3-pip-wheels transformers
    # !python -m pip install --no-index --find-links=../input/fb3-pip-wheels tokenizers
    pass

from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu' # todo
try:
    import kaggle_secrets

    IS_KAGGLE = True
except:
    IS_KAGGLE = False

if IS_KAGGLE:
    FB3_PATH = '../input/....'
else:
    FB3_PATH = '/content/drive/MyDrive/kaggle/fb3'
    FB3_PATH = 'data/comp-data'




# %% papermill={"duration": 2.436763, "end_time": "2022-09-09T15:49:43.027095", "exception": false, "start_time": "2022-09-09T15:49:40.590332", "status": "completed"} tags=[]
anony = None

if CFG.wandb:

    import wandb

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        secret_value_0 = user_secrets.get_secret("wandb_api")
        wandb.login(key=secret_value_0)
        anony = None
    except:
        anony = "must"
        print(
            'If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
        wandb.login(key='adc8abc0714ba20c3a534b907b9d6beec640f847')


def class2dict(f):
    return dict((name, getattr(f, name)) for name in dir(f) if not name.startswith('__'))


# %% [markdown] papermill={"duration": 0.008101, "end_time": "2022-09-09T15:50:01.556438", "exception": false, "start_time": "2022-09-09T15:50:01.548337", "status": "completed"} tags=[]
# # Utils

# %% papermill={"duration": 0.024326, "end_time": "2022-09-09T15:50:01.589144", "exception": false, "start_time": "2022-09-09T15:50:01.564818", "status": "completed"} tags=[]
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores


def get_logger(filename='./' + 'train'):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(asctime)s %(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


LOGGER = get_logger()

# %% [markdown] papermill={"duration": 0.008379, "end_time": "2022-09-09T15:50:01.605740", "exception": false, "start_time": "2022-09-09T15:50:01.597361", "status": "completed"} tags=[]
# # Data Loading

# %% papermill={"duration": 0.136626, "end_time": "2022-09-09T15:50:01.750587", "exception": false, "start_time": "2022-09-09T15:50:01.613961", "status": "completed"} tags=[]
df_train = pd.read_csv(f'{FB3_PATH}/train.csv')
df_test = pd.read_csv(f'{FB3_PATH}/test.csv')
df_submission = pd.read_csv(f'{FB3_PATH}/sample_submission.csv')

print(f"train.shape: {df_train.shape}")
df_train.head()

# %%
print(f"test.shape: {df_test.shape}")
(df_test.head())


# %%
print(f"submission.shape: {df_submission.shape}")
df_submission.head()

# %%
df_sent = df_train.cohesion.value_counts().to_frame('cohesion')
for c in ['syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']:
    df_sent = df_sent.join(df_train[c].value_counts())
df_sent = df_sent.reindex(df_sent.index.sort_values())
df_sent.plot.bar(figsize=(20, 5), title='Distribution of unique scores per axis')

# %% [markdown] papermill={"duration": 0.008867, "end_time": "2022-09-09T15:50:01.768749", "exception": false, "start_time": "2022-09-09T15:50:01.759882", "status": "completed"} tags=[]
# # CV split

# %% papermill={"duration": 0.140203, "end_time": "2022-09-09T15:50:01.917878", "exception": false, "start_time": "2022-09-09T15:50:01.777675", "status": "completed"} tags=[]
kfold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(kfold.split(df_train, df_train[CFG.target_cols])):
    df_train.loc[val_index, 'fold'] = int(n)
df_train['fold'] = df_train['fold'].astype(int)
df_train.groupby('fold').size()

# %% [markdown] papermill={"duration": 0.00927, "end_time": "2022-09-09T15:50:08.617836", "exception": false, "start_time": "2022-09-09T15:50:08.608566", "status": "completed"} tags=[]
# # Dataset

# %%
import os

if os.path.exists('./tokenizer'):
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer')
else:
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained('./' + 'tokenizer/')
CFG.tokenizer = tokenizer


# %% papermill={"duration": 5.779847, "end_time": "2022-09-09T15:50:14.407211", "exception": false, "start_time": "2022-09-09T15:50:08.627364", "status": "completed"} tags=[]
# Define max_len

CFG.max_len = 1429

"""

lengths = []
tk0 = tqdm(df_train['full_text'].fillna("").values, total=len(df_train))
for text in tk0:
    length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
    lengths.append(length)
CFG.max_len = max(lengths) + 3  # cls & sep & sep
LOGGER.info(f"max_len: {CFG.max_len}")
"""

# %%
from io import StringIO
import re
import pandas as pd

NAMES = pd.read_csv(StringIO("""1	Emma	Aidan
2	Emily	Jacob
3	Madison	Ethan
4	Kaitlyn	Nicholas
5	Sophia	Matthew
6	Isabella	Ryan
7	Olivia	Tyler
8	Hannah	Jack
9	Makayla	Joshua
10	Ava	Andrew
11	Abigail	Dylan
12	Sarah	Michael
13	Hailey	Connor"""), sep="\t", header=None).values[:, 1:].flatten()
NAMES

# %%
import numpy as np

np.random.choice(NAMES)

# %%
CITIES = pd.read_csv(StringIO("""1	New York	New York	8,467,513	8,804,190	−3.82%	300.5 sq mi	778.3 km2	29,298/sq mi	11,312/km2	40.66°N 73.93°W
2	Los Angeles	California	3,849,297	3,898,747	−1.27%	469.5 sq mi	1,216.0 km2	8,304/sq mi	3,206/km2	34.01°N 118.41°W
3	Chicago	Illinois	2,696,555	2,746,388	−1.81%	227.7 sq mi	589.7 km2	12,061/sq mi	4,657/km2	41.83°N 87.68°W
4	Houston	Texas	2,288,250	2,304,580	−0.71%	640.4 sq mi	1,658.6 km2	3,599/sq mi	1,390/km2	29.78°N 95.39°W
5	Phoenix	Arizona	1,624,569	1,608,139	+1.02%	518.0 sq mi	1,341.6 km2	3,105/sq mi	1,199/km2	33.57°N 112.09°W
6	Philadelphia	Pennsylvania	1,576,251	1,603,797	−1.72%	134.4 sq mi	348.1 km2	11,933/sq mi	4,607/km2	40.00°N 75.13°W
7	San Antonio	Texas	1,451,853	1,434,625	+1.20%	498.8 sq mi	1,291.9 km2	2,876/sq mi	1,110/km2	29.47°N 98.52°W
8	San Diego	California	1,381,611	1,386,932	−0.38%	325.9 sq mi	844.1 km2	4,256/sq mi	1,643/km2	32.81°N 117.13°W
9	Dallas	Texas	1,288,457	1,304,379	−1.22%	339.6 sq mi	879.6 km2	3,841/sq mi	1,483/km2	32.79°N 96.76°W
10	San Jose	California	983,489	1,013,240	−2.94%	178.3 sq mi	461.8 km2	5,683/sq mi	2,194/km2	37.29°N 121.81°W
11	Austin	Texas	964,177	961,855	+0.24%	319.9 sq mi	828.5 km2	3,007/sq mi	1,161/km2	30.30°N 97.75°W
12	Jacksonville	Florida	954,614	949,611	+0.53%	747.3 sq mi	1,935.5 km2	1,271/sq mi	491/km2	30.33°N 81.66°W
13	Fort Worth	Texas	935,508	918,915	+1.81%	342.9 sq mi	888.1 km2	2,646/sq mi	1,022/km2	32.78°N 97.34°W
14	Columbus	Ohio	906,528	905,748	+0.09%	220.0 sq mi	569.8 km2	4,117/sq mi	1,590/km2	39.98°N 82.98°W
15	Indianapolis	Indiana	882,039	887,642	−0.63%	361.6 sq mi	936.5 km2	2,455/sq mi	948/km2	39.77°N 86.14°W
16	Charlotte	North Carolina	879,709	874,579	+0.59%	308.3 sq mi	798.5 km2	2,837/sq mi	1,095/km2	35.20°N 80.83°W"""),
                     sep="\t", header=None).values[:, 1]
CITIES


# %% papermill={"duration": 0.033181, "end_time": "2022-09-09T15:50:14.450580", "exception": false, "start_time": "2022-09-09T15:50:14.417399", "status": "completed"} tags=[]
class TrainDataset(Dataset):
    def __init__(self, cfg: CFG, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        s = self.texts[item]
        if self.cfg.unscrew_names:
            while True:
                s, cnt = re.subn('Generic_Name', np.random.choice(NAMES), s, count=1)
                if cnt == 0:
                    break
            re.sub('Generic_School', 'school', self.texts[item])
            while True:
                s, cnt = re.subn('Generic_City', np.random.choice(CITIES), s, count=1)
                if cnt == 0:
                    break

        inputs = self.cfg.tokenizer.encode_plus(
            s,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.cfg.max_len,
            pad_to_max_length=True,
            truncation=True
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        label = torch.tensor(self.labels[item], dtype=torch.float32)
        return inputs, label


def collate_fn(inputs):
    inputs, y = default_collate(inputs)
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs, y


# Quick test
if CFG.debug:
    ds = TrainDataset(CFG, df_train)
    ds[0]
    dl = DataLoader(ds, 2, collate_fn=collate_fn)
    next(iter(dl))

# %% [markdown] papermill={"duration": 0.0094, "end_time": "2022-09-09T15:50:14.470068", "exception": false, "start_time": "2022-09-09T15:50:14.460668", "status": "completed"} tags=[]
# # Model

# %%
from random import random
from torch.nn import LSTM


class MeanPooling(torch.nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class LSTMPooling(nn.Module):
    def __init__(self, input_size, dropout, skip_connection):
        super(LSTMPooling, self).__init__()
        self.lstm = LSTM(
            input_size=input_size, hidden_size=input_size // 2, num_layers=1, batch_first=True, bidirectional=True,
            dropout=dropout
        )
        self.skip_connection = skip_connection
        self.mean_pool = MeanPooling()

    def forward(self, last_hidden_state, attention_mask):
        mean_pool = self.mean_pool.forward(last_hidden_state, attention_mask)
        out = []
        for seq, seq_len, mp in zip(last_hidden_state, attention_mask.sum(dim=1).to(int), mean_pool):
            inp = seq[:seq_len.item()].unsqueeze(0)
            output, (h_n, c_n) = self.lstm(inp)
            lstm_out = torch.concat([h_n[0], h_n[-1]], dim=1)
            if self.skip_connection:
                out.append(lstm_out + mp.unsqueeze(0))
            else:
                out.append(lstm_out)
        return torch.concat(out).float()



def gen_attention_mask(batch, seq):
    np.random.seed(42)
    mask = []
    for i in range(batch):
        l = np.random.randint(1, seq, 1)[0]
        mask.append([1.] * l + [0.] * (seq - l))
    return torch.as_tensor(mask)


# Quick test
lstm = LSTMPooling(10, 0., False)
lstm.forward(torch.randn(2, 5, 10), gen_attention_mask(2, 5))

# %%
MeanPooling().forward(torch.randn(2, 5, 10), gen_attention_mask(2, 5)).shape


# %% papermill={"duration": 18.918037, "end_time": "2022-09-09T15:50:33.397956", "exception": false, "start_time": "2022-09-09T15:50:14.479919", "status": "completed"} tags=[]
class FB3ClassifierModel(nn.Module):
    def __init__(self, cfg: CFG, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
            LOGGER.info(self.config)
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        else:
            self.model = AutoModel(self.config)
        if self.cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.cfg.frozen_backbone:
            for p in self.model.parameters():
                p.requires_grad = False

        if self.cfg.emb_type == 'mean':
            self.pool = MeanPooling()
        elif self.cfg.emb_type == 'lstm':
            self.pool = LSTMPooling(input_size=self.config.hidden_size, dropout=cfg.lstm_dropout,
                                    skip_connection=cfg.skip_connection)
        self.fc_regression = nn.Linear(self.config.hidden_size, len(self.cfg.target_cols))
        self._init_weights(self.fc_regression)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        model_output = self.model(**inputs)
        emb = self.pool(model_output[0], inputs['attention_mask'])
        return self.fc_regression(emb)


# Quick test
FB3ClassifierModel(CFG, config_path=None, pretrained=True)


# %% [markdown] papermill={"duration": 0.010169, "end_time": "2022-09-09T15:50:33.448803", "exception": false, "start_time": "2022-09-09T15:50:33.438634", "status": "completed"} tags=[]
# # Loss

# %% papermill={"duration": 0.057624, "end_time": "2022-09-09T15:50:33.517078", "exception": false, "start_time": "2022-09-09T15:50:33.459454", "status": "completed"} tags=[]
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred.shape = (Batch, Column)
        y_true.shape = (Batch, Column)
        """
        loss = nn.functional.mse_loss(y_pred, y_true, reduction='none')
        loss = torch.sqrt(torch.mean(loss, dim=0))
        return torch.mean(loss).to(torch.float32)


# Quick test
loss = RMSELoss()

scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)

lstm = LSTMPooling(10, 0., True).to(DEVICE)

with torch.cuda.amp.autocast(enabled=True):
    pred = lstm.forward(torch.randn(2, 5, 10).to(DEVICE), gen_attention_mask(2, 5).to(DEVICE))
    targets = torch.randn(2, 10).to(DEVICE)
    l = loss(pred, targets)

scaler.scale(l).backward()



# %% papermill={"duration": 0.029801, "end_time": "2022-09-09T15:50:33.557859", "exception": false, "start_time": "2022-09-09T15:50:33.528058", "status": "completed"} tags=[]
class CrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        y_pred.shape = (Batch, Column, ClassProbability)
        y_true.shape = (Batch, Column)
        """
        y_true = (y_true * 2).to(torch.long)  # 1..5 -> 2..10
        loss = torch.stack([nn.functional.cross_entropy(p, t) for p, t in zip(y_pred, y_true)])
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


# Quick test
loss = CrossEntropyLoss(reduction='none')
loss.forward(
    torch.randn(2, 6, 11), torch.randint(1, 10, (2, 6)) / 2
)


# %% [markdown]
# # Training

# %%
def train_fn(cfg: CFG, train_loader, model: FB3ClassifierModel, criterion, optimizer, scheduler, device, run, verbose=True):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = []
    with tqdm(train_loader, desc='Train', disable=not verbose) as progress:
        for step, (inputs, labels) in enumerate(progress):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast(enabled=cfg.apex):
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)
                losses.append(loss.item())
                if cfg.gradient_accumulation_steps > 1:
                    loss = (loss / cfg.gradient_accumulation_steps)
            if run is not None:
                run.log({f"train_loss": losses[-1],
                         f"lr": scheduler.get_last_lr()[0]})
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), CFG.max_grad)
            if (step + 1) % cfg.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if cfg.batch_scheduler:
                    scheduler.step()

    return np.mean(losses)



# %%
def valid_fn(valid_loader, model: FB3ClassifierModel, criterion, device, verbose=True):
    model.eval()
    preds = []
    losses = []
    with tqdm(valid_loader, desc='Eval', disable=not verbose) as progress:
        for step, (inputs, labels) in enumerate(progress):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                out = model.forward(inputs)
                loss = criterion(out, labels)
                losses.append(loss.item())
                preds.append(out.to('cpu').numpy())

    predictions = np.concatenate(preds)
    return np.mean(losses), predictions


# %% papermill={"duration": 0.033556, "end_time": "2022-09-09T15:50:33.690792", "exception": false, "start_time": "2022-09-09T15:50:33.657236", "status": "completed"} tags=[]
def train_loop(cfg: CFG, name, df_folds=None, fold=None, df_train_folds=None, df_valid_fold=None, run=None,
               save_model=True, verbose=True):
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ********* loader **********
    if fold is not None:
        df_train_folds = df_folds[df_folds['fold'] != fold].reset_index(drop=True)
        df_valid_fold = df_folds[df_folds['fold'] == fold].reset_index(drop=True)

    ds_train = TrainDataset(cfg, df_train_folds)
    ds_val = TrainDataset(cfg, df_valid_fold)

    dl_train = DataLoader(ds_train,
                          collate_fn=collate_fn,
                          batch_size=cfg.batch_size,
                          shuffle=True,
                          num_workers=cfg.num_workers,
                          pin_memory=True,
                          drop_last=True)
    dl_val = DataLoader(ds_val,
                        collate_fn=collate_fn,
                        batch_size=cfg.batch_size * 2,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        pin_memory=True,
                        drop_last=False)

    # *********** model & optimizer ***********
    model = FB3ClassifierModel(cfg, config_path=None, pretrained=True)
    torch.save(model.config, 'config.pth')
    model.to(DEVICE)

    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay, head_weight_decay):
        no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}
        optimizer_parameters = [
            # backbone
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},

            # head
            {'params': [p for n, p in model.named_parameters() if "model" not in n and not any(nd in n for nd in no_decay)],
             'lr': decoder_lr, 'weight_decay': head_weight_decay},
            {'params': [p for n, p in model.named_parameters() if "model" not in n and any(nd in n for nd in no_decay)],
             'lr': decoder_lr, 'weight_decay': 0.}
        ]
        return optimizer_parameters

    opt_params = get_optimizer_params(model, encoder_lr=cfg.encoder_lr, decoder_lr=cfg.decoder_lr,
                                      weight_decay=cfg.weight_decay, head_weight_decay=cfg.head_weight_decay)
    optimizer = AdamW(opt_params, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas)

    # *********** scheduler ***********
    def get_scheduler(cfg: cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.pct_warmup_steps * num_train_steps),
                num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=int(cfg.pct_warmup_steps * num_train_steps),
                num_training_steps=num_train_steps,
                num_cycles=cfg.cosine_num_cycles
            )
        else:
            raise Exception("Scheduler not implemented")
        return scheduler

    num_train_steps = int(len(dl_train) * cfg.epochs / cfg.gradient_accumulation_steps)
    scheduler = get_scheduler(cfg, optimizer, num_train_steps)

    # ********** loop ************
    criterion = RMSELoss()
    best_score = np.inf
    best_val_pred = None
    best_model_file_name = f"{name}_fold{fold}_best.pth"
    for epoch in tqdm(range(cfg.epochs), desc='Epochs'):
        # train
        train_loss = train_fn(cfg, dl_train, model, criterion, optimizer, scheduler, DEVICE, run, verbose)

        # eval
        val_rmse_loss, val_pred = valid_fn(dl_val, model, criterion, DEVICE, verbose)

        # scoring
        val_score, val_col_scores = get_score(df_valid_fold[cfg.target_cols].values, val_pred)

        LOGGER.info(
            f'Epoch {epoch} - train_loss: {train_loss:.4f}  val_rmse_loss: {val_rmse_loss:.4f} val_score: {val_score:.4f}  val_col_scores: {val_col_scores}'
        )

        if run is not None:
            run.log({f"epoch": epoch,
                     f"train_epoch_loss": train_loss,
                     f"val_rmse_loss": val_rmse_loss,
                     f"val_score": val_score})

        if val_score < best_score:
            best_score = val_score
            best_val_pred = val_pred
            LOGGER.info(f'Epoch {epoch} - Saving the best model with score {best_score:.4f}')
            if save_model:
                torch.save({'model': model.state_dict(), 'predictions': val_pred}, best_model_file_name)

    # val_pred = torch.load(best_model_file_name, map_location=torch.device('cpu'))['predictions']
    df_valid_fold[[f"pred_{c}" for c in cfg.target_cols]] = best_val_pred
    torch.cuda.empty_cache()
    gc.collect()
    return df_valid_fold


# %%
def get_result(df_eval, cfg=CFG):
    labels = df_eval[cfg.target_cols].values
    preds = df_eval[[f"pred_{c}" for c in cfg.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')
    return score, scores



# %% papermill={"duration": 11979.697909, "end_time": "2022-09-09T19:10:13.399620", "exception": false, "start_time": "2022-09-09T15:50:33.701711", "status": "completed"} tags=[]

if CFG.train:
    df_eval = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            wandb_name = CFG.model.split('/')[-1]
            with wandb.init(project='FB3-regressor',
                            name=f'lstm-{wandb_name}-{fold}',
                            config=class2dict(CFG),
                            anonymous=anony,
                            mode=("disabled" if not CFG.wandb else None)) as run:
                df_eval_fold = train_loop(CFG, 'lstm-classifier', df_train, fold, run=run)
            df_eval = pd.concat([df_eval, df_eval_fold])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(df_eval_fold)

    df_eval = df_eval.reset_index(drop=True)
    LOGGER.info(f"========== CV ==========")
    get_result(df_eval)
    df_eval.to_pickle('./' + 'oof_df.pkl')


# %% [markdown]
# # Cross validate

# %%
def evaluate(cfg: CFG, model: FB3ClassifierModel, df_data, fold):
    df_valid_fold = df_data[df_data['fold'] == fold].reset_index(drop=True)
    ds_val = TrainDataset(cfg, df_valid_fold)
    dl_val = DataLoader(ds_val,
                        collate_fn=collate_fn,
                        batch_size=cfg.batch_size * 2,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        pin_memory=True,
                        drop_last=False)
    eval_criterion = RMSELoss()
    val_rmse_loss, val_pred = valid_fn(dl_val, model, eval_criterion, DEVICE)
    val_score, val_col_scores = get_score(df_valid_fold[cfg.target_cols].values,
                                          model.predict(preds=torch.as_tensor(val_pred)))
    return val_score, val_pred


if CFG.cross_validation:
    fold_preds = []
    for fold in tqdm(range(CFG.n_fold), desc='Folds'):
        gc.collect()
        torch.cuda.empty_cache()
        model = FB3ClassifierModel(CFG, config_path=None, pretrained=True)
        model.load_state_dict(
            torch.load(f'lstm-classifier_fold{fold}_best.pth', map_location=torch.device('cuda'))['model'])
        model.to(DEVICE)

        score, preds = evaluate(CFG, model, df_train, fold)
        fold_preds.append(preds)
        print(score)

    import scipy

    df_features = pd.DataFrame(
        index=np.concatenate([df_train.query('fold == @fold').index for fold in range(CFG.n_fold)]),
        data=scipy.special.softmax(np.concatenate(fold_preds), axis=-1).reshape((-1, 6 * 11)).tolist()
    )
    df_features.sort_index().to_csv('data/lstm_features.csv', index=False)


# %% [markdown]
# # Optuna

# %%
import torch
total_mem = torch.cuda.get_device_properties('cuda:0').total_memory // 10 ** 9

# %%
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'

# %%
# # rm FB3-syntax.sqlite; optuna create-study --study-name "FB3-syntax" --direction minimize    --storage  sqlite:///FB3-syntax.sqlite


# pip install transformers tokenizers sentencepiece optuna wandb iterative-stratification matplotlib pandas

import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback


optuna_column = 'cohesion'

if __name__ == "__main__":
    if len(sys.argv) == 3:
        DEVICE, optuna_column = sys.argv[1], sys.argv[2]
        print('Using device', DEVICE)
        print('Optimizing for column', optuna_column)
        # batch_size = int(batch_size)
        # print('Using bathc size', batch_size)
        # CFG.batch_size = (batch_size)

project_name = f"FB3-{optuna_column}"
wandb_kwargs = {"project": project_name}
wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

@wandbc.track_in_wandb()
def objective(trial: optuna.Trial):
    cfg = copy.copy(CFG)
    cfg.target_cols = [optuna_column]
    cfg.epochs = trial.suggest_int('epochs', 1, 7)
    cfg.pct_warmup_steps = trial.suggest_float('pct_warmup_steps', 0, 0.3)
    cfg.decoder_lr = trial.suggest_float('decoder_lr', 1e-5, 1e-2, log=True)
    cfg.encoder_lr = trial.suggest_float('encoder_lr', 3e-7, 1e-5, log=True)
    cfg.weight_decay = 0.01
    cfg.head_weight_decay = 0 # trial.suggest_float('head_weight_decay', 1e-10, 1, log=True)
    cfg.skip_connection =  False #rial.suggest_categorical('skip_connection', [True, False])
    cfg.prediction_type = 'regression'  #trial.suggest_categorical('prediction_type', ['regression', 'classification'])
    cfg.scheduler = trial.suggest_categorical('scheduler', ['linear', 'cosine'])
    cfg.model = trial.suggest_categorical('model', ['microsoft/deberta-v3-base', "microsoft/deberta-v3-large"])
    cfg.model = 'microsoft/deberta-v3-large'
    if total_mem == 16:
        if cfg.model == 'microsoft/deberta-v3-base':
            batch_size = 3
        elif cfg.model == 'microsoft/deberta-v3-large':
            batch_size = 1
    elif total_mem > 16:
        if cfg.model == 'microsoft/deberta-v3-base':
            batch_size = 3
        elif cfg.model == 'microsoft/deberta-v3-large':
            batch_size = 1

    print('batch_size', batch_size)
    virtual_batch_size = trial.suggest_int('virtual_batch_size', 1, 64, log=True)
    # virtual_batch_size = 64
    if virtual_batch_size <= batch_size:
        cfg.batch_size = virtual_batch_size
        cfg.gradient_accumulation_steps = 1
    else:
        cfg.batch_size = batch_size
        cfg.gradient_accumulation_steps = virtual_batch_size // cfg.batch_size

    cfg.emb_type = 'mean' # trial.suggest_categorical('emb_type', ['lstm', 'mean'])
    # if cfg.emb_type == 'lstm':
    #     cfg.lstm_dropout = trial.suggest_categorical('lstm_dropout', [0, 0.15, 0.3])

    df_eval_fold = train_loop(cfg, 'FB3-regressor', df_train, fold=0, save_model=False, verbose=False)
    score, _ = get_result(df_eval_fold, cfg)
    return score


if CFG.optuna:
    if os.path.exists(f'{project_name}.sqlite'):
        print(f'loading study from sqlite:///{project_name}.sqlite')
        study = optuna.load_study(study_name=project_name, storage=f'sqlite:///{project_name}.sqlite')
    else:
        print('Creating new in-memory study')
        study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial), n_trials=CFG.optuna_trials, callbacks=[wandbc])


# %% [markdown]
# # Sentence-wise modeling

# %%
def split_string(s):
    v = s.split("\n\n")
    r = []
    for i in v:
        if len(i) < 50:
            continue
        n_seg = int(math.ceil(len(i) / 1000))
        n_len = len(i) // n_seg
        for k in range(n_seg):
            if k == n_seg - 1:
                r.append(i[k * n_len:])
            else:
                r.append(i[k * n_len:(k + 1) * n_len])
            # print(k, n_seg, r[-1])
    return r


df_sent = df_train.join(df_train.full_text.apply(lambda s: split_string(s)).explode(), rsuffix='_sent')
df_sent['sent_len'] = df_sent.full_text_sent.str.len()
df_sent.head()

# %%
df_sent.sent_len.plot.hist(bins=100, figsize=(20, 5))

# %%
plt.figure(figsize=(20, 10))
for i, c in enumerate(CFG.target_cols):
    ax = plt.subplot(3, 2, i + 1)
    ax = df_sent.groupby('full_text').median()[['sent_len', 'syntax']].reset_index().pivot(columns='syntax',
                                                                                           values='sent_len',
                                                                                           index='full_text').plot.box(
        title=c, ax=ax
    )
    ax.set_ylim(0, 2000)

# %%
cfg = copy.copy(CFG)
cfg.epochs = 4

# %%
for fold in range(0, cfg.n_fold):
    with wandb.init(project='FB3-sent-classifier',
                    name=f'deberta-v3-{fold}',
                    config=class2dict(cfg)) as run:
        torch.cuda.empty_cache()
        gc.collect()
        train_loop(cfg, 'lstm-sent-classifier',
                   df_sent.drop(columns=['full_text', 'sent_len']).rename(columns={'full_text_sent': 'full_text'}),
                   fold, run)


# %%
def pred_log_proba(cfg, model, df, device):
    ds_val = TrainDataset(cfg, df)
    dl_val = DataLoader(ds_val,
                        collate_fn=collate_fn,
                        batch_size=cfg.batch_size * 2,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        pin_memory=True,
                        drop_last=False)

    model.eval()
    preds = []
    with tqdm(dl_val, desc='Pred proba') as progress:
        for step, (inputs, labels) in enumerate(progress):
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            with torch.no_grad():
                y_preds = model.predict(inputs, raw=True)
                preds.append(y_preds.to('cpu').numpy())

    predictions = np.concatenate(preds)
    return torch.nn.functional.log_softmax(torch.as_tensor(predictions), dim=-1)


def text_scores(df):
    probas = np.array(df.proba.values.tolist())
    text_proba = np.reshape(np.sum(probas, axis=0), (6, 11))
    text_scores = np.argmax(text_proba, axis=-1) / 2.
    s = pd.Series(index=[f'pred_{c}' for c in CFG.target_cols], data=text_scores)
    return s


scores = []
for fold in range(cfg.n_fold):
    gc.collect()
    torch.cuda.empty_cache()
    model = FB3ClassifierModel(cfg, config_path=None, pretrained=True)
    model.load_state_dict(
        torch.load(f'microsoft-deberta-v3-base_fold{fold}_best.pth', map_location=torch.device('cuda'))['model'])
    model.to(DEVICE)

    df_sent_valid = df_sent.query('fold==@fold').copy()
    valid_proba = pred_log_proba(cfg, model, df_sent_valid.drop(columns=['full_text', 'sent_len']).rename(
        columns={'full_text_sent': 'full_text'}), DEVICE)
    df_sent_valid['proba'] = valid_proba.reshape(len(valid_proba), -1).tolist()

    df_valid = df_train.query('fold == @fold').copy()
    df_valid = df_valid.set_index('text_id').join(
        df_sent_valid.groupby('text_id')[['proba']].apply(lambda df: text_scores(df)))
    score, _ = get_result(df_valid)
    scores.append(score)


# %%
np.mean(scores)
