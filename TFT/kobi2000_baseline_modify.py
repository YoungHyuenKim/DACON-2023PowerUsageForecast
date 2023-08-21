# GPU memory usage:
# 23546MiB / 24263MiB
import pprint
# requirements:
# pytorch-forecasting==0.9.0

import sys
import os
import argparse
import shutil
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
# import pytorch_lightning as pl
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, GroupNormalizer
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import (
#     ModelCheckpoint,
#     EarlyStopping,
#     LearningRateMonitor
# )
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting import TemporalFusionTransformer

# category columns
# TODO:: Modifiying
# CATE_COLS = ['num', "mgrp", 'holiday', 'dow', 'cluster', 'hot', 'nelec_cool_flag', 'solar_flag']
CATE_COLS = ['building_number', 'building_type', 'holiday', 'dow', 'cluster', 'hot', 'ess_flag', 'solar_flag']

# TODO:: CLUSTER, 현재 가장 기본적인 클러스터링하여 나온결과.
# building cluster based on kmeans
CLUSTER = {0: [100,
               2,
               3,
               32,
               33,
               34,
               35,
               37,
               38,
               39,
               40,
               41,
               42,
               43,
               44,
               5,
               54,
               6,
               81,
               85,
               86,
               87,
               88,
               89,
               9,
               90,
               91,
               92,
               93,
               94,
               95,
               96,
               97,
               98,
               99],
           1: [1,
               11,
               12,
               16,
               17,
               18,
               19,
               20,
               21,
               22,
               23,
               24,
               25,
               26,
               27,
               28,
               29,
               30,
               31,
               36,
               4,
               45,
               46,
               47,
               48,
               49,
               50,
               51,
               52,
               53,
               55,
               56,
               57,
               58,
               59,
               60,
               69,
               7,
               70,
               71,
               72,
               73,
               74,
               75,
               76,
               77,
               78,
               79,
               8,
               80,
               82,
               83,
               84],
           2: [61, 62, 63, 64, 65, 66, 67, 68],
           3: [10, 13, 14, 15]}

# length of training data for prediction (5 weeks)
ENCODER_LENGTH_IN_WEEKS = 2
device = "gpu"

# learning rate determined by a cv run with train data less 1 trailing week as validation
LRS = [0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.05099279397234306, 0.05099279397234306, 0.05099279397234306, 0.05099279397234306,
       0.005099279397234306, 0.005099279397234306, 0.005099279397234306, 0.005099279397234306,
       0.005099279397234306, 0.005099279397234306, 0.005099279397234306, 0.005099279397234306,
       0.005099279397234306, 0.0005099279397234307, 0.0005099279397234307, 0.0005099279397234307,
       0.0005099279397234307, 0.0005099279397234307, 0.0005099279397234307]

# number of epochs found in cv run
NUM_EPOCHS = 120

# number of seeds to use
START_SEED = 43
NUM_SEEDS = 10
TOP_K = 10
BATCH_SIZE = 128
NUM_WORKERS = 12

# hyper parameters determined by cv runs with train data less 1 trailing week as validation
PARAMS = {
    'gradient_clip_val': 0.8733187217928037,
    'hidden_size': 256,
    'dropout': 0.12544263168386235,
    'hidden_continuous_size': 92,
    'attention_head_size': 4,
    'learning_rate': 0.08
}

parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', nargs='+', type=int, default=list(range(START_SEED, START_SEED + NUM_SEEDS)))
# parser.add_argument('--val', default=False, action='store_true')
# parser.add_argument('--nepochs', '-e', type=int, default=NUM_EPOCHS)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fit', default=True, action='store_true')
parser.add_argument('--forecast', default=True, action='store_true')
parser.add_argument('--dataroot', '-d', type=str, default="../Data/rawData")
args = parser.parse_args()
args.val = False
args.nepochs = NUM_EPOCHS
# print(args)

DATAROOT = Path(args.dataroot)  # 코드에 ‘/data’ 데이터 입/출력 경로 포함
CKPTROOT = DATAROOT / "ckpts"  # directory for model checkpoints
CSVROOT = DATAROOT / "csvs"  # directory for prediction outputs
SUBFN = DATAROOT / "sub.csv"  # final submission file path
LOGDIR = DATAROOT / "logs"  # pytorch_forecasting requirs logger


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def __building_info_prep(df):
    # solar_power_capacity not '-' -> solar_flag on
    df["solar_flag"] = df["solar_power_capacity"] != "-"
    df["solar_power_capacity"] = df.apply(
        lambda row: 0 if row["solar_power_capacity"] == "-" else row["solar_power_capacity"], axis=1)

    # ess_capactiy not '-' -> ess_flag on
    df["ess_flag"] = df["ess_capacity"] != "-"
    df["ess_capacity"] = df.apply(
        lambda row: 0 if row["ess_capacity"] == "-" else row["ess_capacity"], axis=1)
    df["pcs_capacity"] = df.apply(
        lambda row: 0 if row["pcs_capacity"] == "-" else row["pcs_capacity"], axis=1)
    return df


# prepare data features
def __date_prep(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['dow'] = df['date_time'].dt.weekday
    df['date'] = df['date_time'].dt.date.astype('str')
    df['day'] = df['date_time'].dt.day
    df['month'] = df['date_time'].dt.month

    # FEATURE: saturday, sunday and speical holidays flagged as `holiday` flag
    # TODO:: Check Special_days..
    # 0601 : 전국동시지방선거
    # 6 6 현층일
    # 815 광복절
    special_days = ['2022-06-01', '2022-06-06', '2022-08-15']
    df['holiday'] = df['dow'].isin([5, 6]).astype(int)
    df.loc[df.date.isin(special_days), 'holiday'] = 1

    # FEATURE: `hot` flag when the next day is holiday
    hot = df.groupby('date').first()['holiday'].shift(-1).fillna(0).astype(int)
    hot = hot.to_frame().reset_index().rename({'holiday': "hot"}, axis=1)
    df = df.merge(hot, on='date', how='left')

    # FEATURE: `cumhol` - how many days left in 연휴
    h = (df.groupby('date').first()['holiday'] != 0).iloc[::-1]
    df1 = h.cumsum() - h.cumsum().where(~h).ffill().fillna(0).astype(int).iloc[::-1]
    df1 = df1.to_frame().reset_index().rename({'holiday': "cumhol"}, axis=1)
    df = df.merge(df1, on='date', how='left')

    return df


# read data, process date and assign cluster number
def __read_df():
    columns = {
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'target'
    }
    building_columns = {
        '건물번호': 'building_number',
        '건물유형': 'building_type',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'solar_power_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity'
    }
    translation_dict = {
        '건물기타': 'Other_Buildings',
        '공공': 'Public',
        '대학교': 'University',
        '데이터센터': 'Data_Center',
        '백화점및아울렛': 'Department_Store_and_Outlet',
        '병원': 'Hospital',
        '상용': 'Commercial',
        '아파트': 'Apartment',
        '연구소': 'Research_Institute',
        '지식산업센터': 'Knowledge_Industry_Center',
        '할인마트': 'Discount_Mart',
        '호텔및리조트': 'Hotel_and_Resort'
    }

    train_df = pd.read_csv(DATAROOT / 'train.csv', encoding='utf-8')
    train_df = train_df.rename(columns=columns)
    test_df = pd.read_csv(DATAROOT / 'test.csv', encoding='utf-8')
    test_df = test_df.rename(columns=columns)

    building_df = pd.read_csv(DATAROOT / 'building_info.csv', encoding="utf-8")
    building_df = building_df.rename(columns=building_columns)
    building_df['building_type'] = building_df['building_type'].replace(translation_dict)
    building_df = __building_info_prep(building_df)

    train_df = pd.merge(train_df, building_df, on='building_number', how='left')
    test_df = pd.merge(test_df, building_df, on='building_number', how='left')

    __sz = train_df.shape[0]

    df = pd.concat([train_df, test_df])

    # assing cluster number to building
    for k, nums in CLUSTER.items():
        df.loc[df.building_number.isin(nums), 'cluster'] = k

    df = __date_prep(df)

    return df.iloc[:__sz].copy(), df.iloc[__sz:].copy()


# add aggregate(mean) target feature for 'cluster', 'building', 'mgrp' per date
# TODO:: Avoid Data Leaking
def add_feats(df):
    df.reset_index(drop=True, inplace=True)

    cols = ['target']
    stats = ['mean']

    # target null in test set to null for other columns care must be taken
    g = df.groupby(['date', 'cluster'])
    for s in stats:
        col_mapper = {c: f"{s}_{c}_cluster" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    g = df.groupby(['date', 'building_number'])
    for s in stats:
        col_mapper = {c: f"{s}_{c}_building_number" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    # g = df.groupby(['date', 'mgrp'])
    # for s in stats:
    #     col_mapper = {c: f"{s}_{c}_mgrp" for c in cols}
    #     tr = g[cols].transform(s).rename(col_mapper, axis=1)
    #     df = pd.concat([df, tr], axis=1)

    g = df.groupby(['date'])
    for s in stats:
        col_mapper = {c: f"{s}_{c}" for c in cols}
        tr = g[cols].transform(s).rename(col_mapper, axis=1)
        df = pd.concat([df, tr], axis=1)

    # Discomfort index
    tr = (0.81 * df["temperature"] + 0.01 * df["humidity"] + 46.3).rename("Discomfort_index")
    df = pd.concat([df, tr], axis=1)

    return df


# interpolate NA values in test dataset
def interpolate_(df):
    # https://dacon.io/competitions/official/235736/codeshare/2844?page=1&dtype=recent
    # 에서 제안된 방법으로
    __methods = {
        'temperature': 'quadratic',
        'windspeed': 'linear',
        'humidity': 'quadratic',
        'rainfall': 'linear',
    }
    # TODO:: group building_number interpolate...
    for col, method in __methods.items():
        df[col] = df[col].interpolate(method=method)
        if method == 'quadratic':
            df[col] = df[col].interpolate(method='linear')
        df[col] = df[col].fillna(0)


# prepare train and test data
def prep():
    train_df, test_df = __read_df()

    # get nelec_cool_flag and solar_flag from training data
    test_df = test_df.drop(['solar_radiation', 'sunshine'], axis=1)
    train_df = train_df.drop(['solar_radiation', 'sunshine'], axis=1)
    # test_df = test_df.merge(train_df.groupby("building_number").first()[['ess_flag', 'solar_flag']].reset_index(), on="building_number",
    #                         how="left")

    # interpolate na in test_df for temperature, windspeed, humidity, precipitation & insolation
    interpolate_(train_df)
    interpolate_(test_df)

    # Not Using...
    # FEATURE(mgrp): group buildings having same temperature and windspeed measurements
    # s = train_df[train_df.date_time == '2022-06-01 00:00:00'].groupby(['temperature', 'windspeed']).ngroup()
    # s.name = 'mgrp'
    # mgrps = train_df[['building_number']].join(s, how='inner')

    df = pd.concat([train_df, test_df])
    # df = df.merge(mgrps, on='building_number', how='left')
    sz = train_df.shape[0]

    # add aggregate target features
    df = add_feats(df)

    # add log target
    df["log_target"] = np.log(df.target + 1e-8)

    for col in CATE_COLS:
        df[col] = df[col].astype(str).astype('category')

    # add time index feature
    __ix = df.columns.get_loc('date_time')
    df['time_idx'] = df["date_time"].dt.dayofyear * 24 + df['date_time'].dt.hour

    train_df = df.iloc[:sz].copy()
    test_df = df.iloc[sz:].copy()

    return train_df, test_df


# build traind datset
def load_dataset(train_df, validate=False):
    max_encoder_length = 24 * 7 * ENCODER_LENGTH_IN_WEEKS  # use 5 past weeks
    max_prediction_length = 24 * 7  # to predict 1 week of future
    training_cutoff = train_df["time_idx"].max() - max_prediction_length

    # build training dataset
    tr_ds = TimeSeriesDataSet(
        # with validate=False use all data
        train_df[lambda x: x.time_idx <= training_cutoff] if validate else train_df,
        time_idx="time_idx",
        target="target",
        group_ids=["building_number"],
        min_encoder_length=1,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=CATE_COLS,
        static_categoricals=["building_number", "cluster"],
        time_varying_known_reals=[
            "time_idx",
            'hour',
            "temperature",
            "rainfall",
            "windspeed",
            "humidity",
            'cumhol',
            'total_area',
            "ess_capacity",
            "pcs_capacity",
            "Discomfort_index"
        ],
        target_normalizer=GroupNormalizer(groups=["building_number"], transformation="softplus"),
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "target",
            "log_target",
            "mean_target",
            "mean_target_building_number",  # Check
            # "mean_target_mgrp",
            "mean_target_cluster"
        ],
        add_relative_time_idx=True,  # add as feature
        add_target_scales=True,  # add as feature
        add_encoder_length=True,  # add as feature
    )
    va_ds = None
    if validate:
        # validation dataset not used for submission
        va_ds = TimeSeriesDataSet.from_dataset(
            tr_ds, train_df, predict=True, stop_randomization=True
        )

    return tr_ds, va_ds


# training
def fit(seed, tr_ds, va_loader=None, find_check_points=False):
    seed_all(seed)  # doesn't really work as training is non-deterministic

    # create dataloaders for model
    tr_loader = tr_ds.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    if va_loader is not None:
        # stop training, when loss metric does not improve on validation set
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=20,
            verbose=True,
            mode="min"
        )
        lr_logger = LearningRateMonitor(logging_interval="epoch")  # log the learning rate
        callbacks = [lr_logger, early_stopping_callback]
    else:
        # gather 10 checkpoints with best traing loss
        early_stopping_callback = EarlyStopping(
            monitor="train_loss",
            min_delta=1e-4,
            patience=20,
            verbose=True,
            mode="min"
        )

        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=CKPTROOT,
            filename=f'seed={seed}' + '-{epoch:03d}-{train_loss:.5f}',
            save_top_k=TOP_K
        )
        callbacks = [checkpoint_callback, early_stopping_callback]

    # create trainer
    trainer = pl.Trainer(
        max_epochs=args.nepochs,
        accelerator=device,
        gradient_clip_val=PARAMS['gradient_clip_val'],
        limit_train_batches=30,
        callbacks=callbacks,
        logger=TensorBoardLogger(LOGDIR)
    )

    # use pre-deterined leraning rate schedule for final submission
    learning_rate = LRS if va_loader is None else PARAMS['learning_rate']

    # initialise model with pre-determined hyperparameters
    tft = TemporalFusionTransformer.from_dataset(
        tr_ds,
        learning_rate=learning_rate,
        hidden_size=PARAMS['hidden_size'],
        attention_head_size=PARAMS['attention_head_size'],
        dropout=PARAMS['dropout'],
        hidden_continuous_size=PARAMS['hidden_continuous_size'],
        output_size=1,
        loss=SMAPE(),  # SMAPE loss
        log_interval=10,  # log example every 10 batches
        logging_metrics=[SMAPE()],
        reduce_on_plateau_patience=4,  # reduce learning automatically
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")

    kwargs = {'train_dataloaders': tr_loader}
    if va_loader:
        kwargs['val_dataloaders'] = va_loader
    if find_check_points:
        seed_check_points = CKPTROOT.glob(f"0816_epoch_60/seed={seed}-*.ckpt")
        min_loss = 10000000000
        kwargs["ckpt_path"] = None
        for check_point in seed_check_points:
            loss = float(check_point.name.split("=")[-1][:-5])
            if loss < min_loss:
                min_loss = loss
                kwargs["ckpt_path"] = check_point
        print(f"Fit : ckpt_path", kwargs["ckpt_path"])
    # fit network
    trainer.fit(
        tft,
        **kwargs
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"{best_model_path=}")
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return best_tft


# predict 1 week
def forecast(ckpt, train_df, test_df):
    # load model
    best_tft = TemporalFusionTransformer.load_from_checkpoint(ckpt)
    max_encoder_length = best_tft.dataset_parameters['max_encoder_length']
    max_prediction_length = best_tft.dataset_parameters['max_prediction_length']

    assert max_encoder_length == ENCODER_LENGTH_IN_WEEKS * 24 * 7 and max_prediction_length == 1 * 24 * 7

    # use 5 weeks of training data at the end
    encoder_data = train_df[lambda x: x.time_idx > x.time_idx.max() - max_encoder_length]

    # get last entry from training data
    last_data = train_df.iloc[[-1]]

    # fill NA target value in test data with last values from the train dataset
    target_cols = [c for c in test_df.columns if 'target' in c]
    for c in target_cols:
        test_df.loc[:, c] = last_data[c].item()

    decoder_data = test_df

    # combine encoder and decoder data. decoder data is to be predicted
    new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
    # new_raw_predictions, new_x = best_tft.predict(new_prediction_data, mode="raw", return_x=True)
    new_raw_predictions = best_tft.predict(new_prediction_data, mode="raw", return_x=True)

    # num_labels: mapping from 'num' categorical feature to index in new_raw_predictions['prediction']
    #             {'5': 4, '6': 6, ...}
    # new_raw_predictions['prediction'].shape = (60, 168, 1)
    num_labels = best_tft.dataset_parameters['categorical_encoders']['building_number'].classes_

    # preds = new_raw_predictions['prediction'].squeeze()
    preds = new_raw_predictions.output.prediction.squeeze().cpu()

    sub_df = pd.read_csv(DATAROOT / "sample_submission.csv")

    # get prediction for each building (num)
    for n, ix in num_labels.items():
        sub_df.loc[sub_df.num_date_time.str.startswith(f"{n}_"), 'answer'] = preds[ix].numpy()

    # save predction to a csv file
    outfn = CSVROOT / (Path(ckpt).stem + '.csv')
    print(outfn)
    sub_df.to_csv(outfn, index=False)


def ensemble(outfn):
    # get all prediction csv files
    fns = list(CSVROOT.glob("*.csv"))
    df0 = pd.read_csv(fns[0])
    df = pd.concat([df0] + [pd.read_csv(fn).loc[:, 'answer'] for fn in fns[1:]], axis=1)
    # get median of all predcitions
    df['median'] = df.iloc[:, 1:].median(axis=1)
    df = df[['num_date_time', 'median']]
    df = df.rename({'median': 'answer'}, axis=1)
    # save to submission file
    df.to_csv(outfn, index=False)


# not used for final submission
def validate(seed, tr_ds, va_ds):
    va_loader = va_ds.to_dataloader(
        train=False, batch_size=BATCH_SIZE * 10, num_workers=NUM_WORKERS
    )
    best_tft = fit(seed, tr_ds, va_loader)
    actuals = torch.cat([y[0] for x, y in iter(va_loader)])
    predictions = best_tft.predict(va_loader)
    smape_per_num = SMAPE(reduction="none")(predictions, actuals).mean(1)
    print(smape_per_num)
    print(smape_per_num.mean())


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def __group_by_building_(data):
    by_weekday = data.groupby(["building_number", "dow"])['target'].median().reset_index()
    by_weekday = by_weekday.pivot(index="building_number", columns="dow", values="target").reset_index()
    by_weekday = by_weekday.rename(columns=dict([(str(i), f"weekday_{i}") for i in range(7)]))
    by_hour = data.groupby(["building_number", "hour"])['target'].median().reset_index()
    by_hour = by_hour.pivot(index="building_number", columns="hour", values="target").reset_index()
    df = pd.merge(by_weekday, by_hour, how="right", on="building_number")
    # df = df.set_index("building_number")
    return df


def __scaling_for_cluster(df):
    for i in range(len(df)):
        # 요일 별 전력 중앙값에 대해 scaling
        df.iloc[i, 1:8] = (df.iloc[i, 1:8] - df.iloc[i, 1:8].mean()) / df.iloc[i, 1:8].std()
        # 시간대별 전력 중앙값에 대해 scaling
        df.iloc[i, 8:] = (df.iloc[i, 8:] - df.iloc[i, 8:].mean()) / df.iloc[i, 8:].std()

    # fig = plt.figure(figsize=(10, 3))
    # for i in range(len(df)):
    #     plt.plot(df.iloc[i, 1:8], alpha=0.5, linewidth=0.5)
    # # plt.show()
    # fig = plt.figure(figsize=(20, 3))
    # for i in range(len(df)):
    #     plt.plot(df.iloc[i, 8:], alpha=0.5, linewidth=0.5)
    # plt.show()

    return df


def change_n_clusters(n_clusters, data, visualize=False):
    data = __group_by_building_(data)
    data = __scaling_for_cluster(data)
    data.columns = data.columns.astype('str')

    sum_of_squared_distance = []
    for n_cluster in n_clusters:
        kmeans = KMeans(n_clusters=n_cluster, init="k-means++", random_state=22)
        kmeans.fit(data.iloc[:, 1:])

        sum_of_squared_distance.append(kmeans.inertia_)
        if visualize:
            km_cluster = kmeans.fit_predict(data.iloc[:, 1:])

            cluster = {}
            for i, c in enumerate(km_cluster):
                if c not in cluster:
                    cluster[c] = []
                cluster[c].append(int(data.loc[i, "building_number"]))
            print(n_cluster)
            pprint.pprint(cluster)

            data["km_cluster"] = km_cluster
            visualize_kmeans_cluster_plot(data)
            visualize_kmeans_cluster_heatmap(data)

    plt.figure(1, figsize=(8, 5))
    plt.plot(n_clusters, sum_of_squared_distance, 'o')
    plt.plot(n_clusters, sum_of_squared_distance, '-', alpha=0.5)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.show()
    plt.pause(100)


def visualize_kmeans_cluster_plot(df_clust):
    n_c = len(np.unique(df_clust.km_cluster))

    fig = plt.figure(figsize=(20, 8))
    for c in range(n_c):
        temp = df_clust[df_clust.km_cluster == c]
        plt.subplot(n_c, 2, 2 * c + 1)
        for i in range(len(temp)):
            plt.plot(temp.iloc[i, 1:8], linewidth=0.7, )
            plt.title(f'cluster{c}')
            plt.xlabel('')
            plt.xticks([])
        plt.subplot(n_c, 2, 2 * c + 2)
        for i in range(len(temp)):
            plt.plot(temp.iloc[i, 8:-6], linewidth=0.7)
            plt.title(f'cluster{c}')
            plt.xlabel('')
            plt.xticks([])
    plt.show()


def visualize_kmeans_cluster_heatmap(df_clust):
    pass


if __name__ == "__main__":
    [p.mkdir(exist_ok=True) for p in (CKPTROOT, CSVROOT, LOGDIR)]

    train_df, test_df = prep()
    tr_ds, va_ds = load_dataset(train_df, args.val)
    torch.set_float32_matmul_precision('medium')

    DO_CLUSTER = False
    if DO_CLUSTER:
        # change_n_clusters([2, 3, 4, 5, 6, 7, 8, 9, 10, 11], train_df)
        change_n_clusters([4], train_df, visualize=True)

    if args.val:
        validate(args.seed[0], tr_ds, va_ds)
    else:
        if args.fit:
            print("### FIT ###")
            for s in args.seed:
                fit(s, tr_ds, va_ds, find_check_points=False)

        if args.forecast:
            print("### FORECAST ###")
            for p in CKPTROOT.glob("*.ckpt"):
                forecast(p, train_df, test_df)

            print("### ENSEMBLING ###")
            ensemble(SUBFN)
