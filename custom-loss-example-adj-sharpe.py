from pathlib import Path
import numpy as np 
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor 
import torch
import csv
from torch.autograd import grad
import example_model as examples
from typing import Tuple

PREDICTION_NAME = examples.PREDICTION_NAME
tournament_data = examples.read_csv("numerai_tournament_data.csv")
train = examples.read_csv("numerai_training_data.csv")

target = examples.TARGET_NAME
feature_columns = [c for c in train if c.startswith("feature")] 

# use example model as an initial model
MODEL_EXAMPLE_FILE = Path("example_model.xgb")

model_init = XGBRegressor(max_depth=5, learning_rate=0.01, n_estimators=2000, colsample_bytree=0.1, nthread=6)
if MODEL_EXAMPLE_FILE.is_file():
    print("Loading pre-trained example model...")
    model_init.load_model(MODEL_EXAMPLE_FILE)
else:
    model_init.fit(train[feature_columns], train[target])
    model_init.save_model(MODEL_EXAMPLE_FILE)

# get prediction from initial model as starting point to improve upon
base_margin = model_init.predict(train[feature_columns])

# get indexes for each era
era_idx = [np.where(train.era==uera)[0] for uera in train.era.unique()]


# define adjusted sharpe in terms of cost adjusted numerai sharpe
def numerai_sharpe(x):
    return (x.mean() -0.010415154) / (x.std()+1)

def skew(x):
    mx = x.mean()
    m2 = ((x-mx)**2).mean()
    m3 = ((x-mx)**3).mean()
    return m3/(m2**1.5)    

def kurtosis(x):
    mx = x.mean()
    m4 = ((x-mx)**4).mean()
    m2 = ((x-mx)**2).mean()
    return (m4/(m2**2))-3

def adj_sharpe(x):
    return numerai_sharpe(x) * (1 + ((skew(x) / 6) * numerai_sharpe(x)) - ((kurtosis(x) / 24) * (numerai_sharpe(x) ** 2)))

# use correlation as the measure of fit
def corr(pred, target):
    pred_n = pred - pred.mean(dim=0)
    pred_n = pred_n / pred_n.norm(dim=0)

    target_n = target - target.mean(dim=0)
    target_n = target_n / target_n.norm(dim=0)
    l = torch.matmul(pred_n, target_n)
    return l

# definte a custom objective for XGBoost
def adj_sharpe_obj(ytrue, ypred):
    # convert to pytorch tensors
    ypred_th = torch.tensor(ypred, requires_grad=True)
    ytrue_th = torch.tensor(ytrue)
    all_corrs = []

    # get correlations in each era
    for ee in era_idx:
        score = corr(ypred_th[ee], ytrue_th[ee])
        all_corrs.append(score)
    
    all_corrs = torch.stack(all_corrs)

    # calculate adjusted sharpe using correlations
    #loss = -all_corrs.mean()
    loss = -adj_sharpe(all_corrs)
    print(f'Current loss:{loss}')

    # calculate gradient and convert to numpy
    loss_grads = grad(loss, ypred_th, create_graph=True)[0]
    loss_grads = loss_grads.detach().numpy()

    # return gradient and ones instead of Hessian diagonal
    return loss_grads, np.ones(loss_grads.shape)

def printModelScore(model, train, val, model2=None):
   # Check the per-era correlations on the training set (in sample)
    if(model2 is None):
        train[PREDICTION_NAME] = model.predict(train[feature_columns])
        val[PREDICTION_NAME] = model.predict(val[feature_columns])
    else:
        train[PREDICTION_NAME] = model.predict(train[feature_columns]) + model2.predict(train[feature_columns])
        val[PREDICTION_NAME] = model.predict(val[feature_columns]) + model2.predict(val[feature_columns])

    train_correlations = train.groupby("era").apply(examples.score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
    print(f"On training the average per-era payout is {examples.payout(train_correlations).mean()}")

    # Check the per-era correlations on the validation set (out of sample)
    validation_correlations = val.groupby("era").apply(examples.score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and "
    f"std {validation_correlations.std(ddof=0)}")
    print(f"On validation the average per-era payout is {examples.payout(validation_correlations).mean()}")

    # Check the "sharpe" ratio on the validation set
    validation_sharpe = validation_correlations.mean() / validation_correlations.std(ddof=0)
    print(f"Validation Sharpe: {validation_sharpe}") 

MODEL_FILE = Path("adj_sharp_model.xgb")

model_adj_sharpe = XGBRegressor(max_depth=5, learning_rate=5, n_estimators=200, nthread=6, colsample_bytree=0.15, objective=adj_sharpe_obj, verbose=True)
validation_data = tournament_data[tournament_data.data_type == "validation"]
if MODEL_FILE.is_file():
    print("Loading pre-trained model...")
    model_adj_sharpe.load_model(MODEL_FILE)
else:
    print("Training model...")
    model_adj_sharpe.fit(
          train[feature_columns], train[target],
          base_margin=base_margin)
    model_adj_sharpe.save_model(MODEL_FILE)


print("---------Init Scores-----------")
printModelScore(model_init, train, validation_data)
print("---------Combined Scores--------------")
printModelScore(model_init, train, validation_data, model_adj_sharpe)

tournament_data[PREDICTION_NAME]= model_init.predict(tournament_data[feature_columns]) + \
            model_adj_sharpe.predict(tournament_data[feature_columns])
tournament_data[PREDICTION_NAME]= examples.unif(tournament_data[PREDICTION_NAME])
tournament_data[PREDICTION_NAME].to_csv("submission.csv", header=True)

    
