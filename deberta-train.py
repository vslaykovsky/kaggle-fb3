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
class CFG:
    wandb = True
    debug = False
    train = False
    optuna = False
    optuna_trials = 30
    cross_validation = False

    scheduler = 'cosine'  # ['linear', 'cosine', 'onecycle']
    cosine_num_cycles = 0.5
    batch_scheduler = True
    pct_warmup_steps = 0.15
    encoder_lr = 8.802e-7  # [5e-6, 8e-6, 9e-6, 1e-5]
    decoder_lr = .00001809
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.01
    head_weight_decay = 0.
    max_grad = 1.

    apex = True
    num_workers = 12
    model = "microsoft/deberta-v3-large"
    epochs = 6
    batch_size = 3
    gradient_checkpointing = True
    gradient_accumulation_steps = 1  # [8, 16, 32]
    max_len = 1429
    all_target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    target_cols = ['cohesion']
    seed = 42
    n_fold = 7
    virtual_batch_size = 3


CUSTOM_CONFIGS = {
    "cohesion": {
        # 0.4754
        "target_cols": ['cohesion'],
        "decoder_lr": 0.00001809,
        "encoder_lr": 8.802e-7,
        "epochs": 6,
        "model": "microsoft/deberta-v3-large",
        "pct_warmup_steps": 0.1453,
        "scheduler": "cosine",
        "virtual_batch_size": 3
    },
    "syntax": {
        # 0.4369
        "target_cols": ['syntax'],
        "decoder_lr": 0.0006275,
        "encoder_lr": 0.000004268,
        "epochs": 3,
        "model": "microsoft/deberta-v3-base",
        "pct_warmup_steps": 0.26,
        "scheduler": "linear",
        "virtual_batch_size": 9
    },
    "vocabulary": {
        # 0.4064
        "target_cols": ['vocabulary'],
        "decoder_lr": 0.00001998,
        "encoder_lr": 0.000001588,
        "epochs": 7,
        "model": "microsoft/deberta-v3-large",
        "pct_warmup_steps": 0.072,
        "scheduler": "cosine",
        "virtual_batch_size": 2
    },
    "phraseology": {
        # 0.4496
        "target_cols": ['phraseology'],
        "decoder_lr": 0.00008505,
        "encoder_lr": 0.000002717,
        "epochs": 6,
        "model": "microsoft/deberta-v3-large",
        "pct_warmup_steps": 0.08377,
        "scheduler": "cosine",
        "virtual_batch_size": 14
    },
    "grammar": {
        # 0.4601
        "target_cols": ['grammar'],
        "decoder_lr": 0.0001013,
        "encoder_lr": 0.000005008,
        "epochs": 2,
        "model": "microsoft/deberta-v3-base",
        "pct_warmup_steps": 0.2172,
        "scheduler": "linear",
        "virtual_batch_size": 5
    },
    "conventions": {
        # 0.4341
        "target_cols": ['conventions'],
        "decoder_lr": 0.0001476,
        "encoder_lr": 0.000007816,
        "epochs": 3,
        "model": "microsoft/deberta-v3-base",
        "pct_warmup_steps": 0.1969,
        "scheduler": "cosine",
        "virtual_batch_size": 3
    },
}


import sys

if len(sys.argv) > 1:
    # from console
    if sys.argv[1] == "train":
        CFG.train = True
    elif sys.argv[1] == "optuna":
        CFG.optuna = True

# %%
import copy
import gc
import warnings
import sys
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm.auto import tqdm
from pprint import pprint

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

TOTAL_CUDA_MEM = torch.cuda.get_device_properties('cuda:0').total_memory // 10 ** 9
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'True'


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
        wandb.login(key=os.environ['WANDB_KEY'])
        # wandb.login(key='adc8abc0714ba20c3a534b907b9d6beec640f847')


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

df_train

# %%
df_test

# %%
df_submission

# %% [markdown] papermill={"duration": 0.008867, "end_time": "2022-09-09T15:50:01.768749", "exception": false, "start_time": "2022-09-09T15:50:01.759882", "status": "completed"} tags=[]
# # CV split

# %% papermill={"duration": 0.140203, "end_time": "2022-09-09T15:50:01.917878", "exception": false, "start_time": "2022-09-09T15:50:01.777675", "status": "completed"} tags=[]
kfold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(kfold.split(df_train, df_train[CFG.all_target_cols])):
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

def gen_attention_mask(batch, seq):
    np.random.seed(42)
    mask = []
    for i in range(batch):
        l = np.random.randint(1, seq, 1)[0]
        mask.append([1.] * l + [0.] * (seq - l))
    return torch.as_tensor(mask)



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

        self.pool = MeanPooling()
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
            if save_model:
                LOGGER.info(f'Epoch {epoch} - Saving the best model with score {best_score:.4f}')
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
    try:
        fold = 0
        if len(sys.argv) == 3:
            # python deberta-train.py train cohesion
            fold = int(sys.argv[2])


        df_eval = pd.DataFrame()
        for column in tqdm(CFG.all_target_cols, desc='Column'):
            CFG.target_cols = [column]

            for k, v in CUSTOM_CONFIGS[column].items():
                setattr(CFG, k, v)

            if CFG.virtual_batch_size <= CFG.batch_size:
                CFG.batch_size = CFG.virtual_batch_size
                CFG.gradient_accumulation_steps = 1
            else:
                CFG.gradient_accumulation_steps = CFG.virtual_batch_size // CFG.batch_size

            print('Training with CFG')
            pprint(vars(CFG))


            wandb_name = CFG.model.split('/')[-1]
            with wandb.init(project=f'FB3-train-{CFG.target_cols[0]}',
                            name=f'{wandb_name}-{column}-{fold}',
                            config=class2dict(CFG),
                            anonymous=anony,
                            mode=("disabled" if not CFG.wandb else None)) as run:
                df_eval_fold = train_loop(CFG, 'regressor', df_train, fold, run=run)
            df_eval = pd.concat([df_eval, df_eval_fold])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(df_eval_fold)

        df_eval = df_eval.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(df_eval)
        df_eval.to_pickle('./' + 'oof_df.pkl')
    finally:
        with open('finished', 'w+') as f:
            f.write(' ')




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
# Use run.sh to create a study


def objective(trial: optuna.Trial, optuna_column):
    cfg = copy.copy(CFG)
    cfg.target_cols = [optuna_column]
    cfg.epochs = trial.suggest_int('epochs', 1, 7)
    cfg.pct_warmup_steps = trial.suggest_float('pct_warmup_steps', 0, 0.3)
    cfg.decoder_lr = trial.suggest_float('decoder_lr', 1e-5, 1e-2, log=True)
    cfg.encoder_lr = trial.suggest_float('encoder_lr', 3e-7, 1e-5, log=True)
    cfg.weight_decay = 0.01
    cfg.scheduler = trial.suggest_categorical('scheduler', ['linear', 'cosine'])
    cfg.model = trial.suggest_categorical('model', ['microsoft/deberta-v3-base', "microsoft/deberta-v3-large"])
    if cfg.model == 'microsoft/deberta-v3-base':
        batch_size = 2
    elif cfg.model == 'microsoft/deberta-v3-large':
        batch_size = 1

    virtual_batch_size = trial.suggest_int('virtual_batch_size', 1, 64, log=True)
    if virtual_batch_size <= batch_size:
        cfg.batch_size = virtual_batch_size
        cfg.gradient_accumulation_steps = 1
    else:
        cfg.batch_size = batch_size
        cfg.gradient_accumulation_steps = virtual_batch_size // cfg.batch_size


    df_eval_fold = train_loop(cfg, 'FB3-regressor', df_train, fold=0, save_model=False, verbose=False)
    score, _ = get_result(df_eval_fold, cfg)
    return score


if CFG.optuna:
    optuna_column = 'cohesion'

    # running from console
    if len(sys.argv) == 4:
        # python deberta-train.py optuna cuda:1 cohesion
        DEVICE, optuna_column = sys.argv[2], sys.argv[3]
    elif len(sys.argv) == 3:
        # python deberta-train.py optuna cohesion
        optuna_column = sys.argv[2]

    print('Using device', DEVICE)
    print('Optimizing for column', optuna_column)

    project_name = f"FB3-{optuna_column}"
    wandb_kwargs = {"project": project_name}
    wandbc = WeightsAndBiasesCallback(wandb_kwargs=wandb_kwargs, as_multirun=True)

    if os.path.exists(f'{project_name}.sqlite'):
        print(f'Loading study from sqlite:///{project_name}.sqlite')
        study = optuna.load_study(study_name=project_name, storage=f'sqlite:///{project_name}.sqlite')
    else:
        print('Creating new in-memory study')
        study = optuna.create_study(direction='minimize')

    study.optimize(lambda trial: objective(trial, optuna_column), n_trials=CFG.optuna_trials, callbacks=[wandbc])
