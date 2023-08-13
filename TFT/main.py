import os
import pathlib
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

from pytorch_forecasting.data.examples import get_stallion_data

device = "gpu"
special_days = [
    "easter_day",
    "good_friday",
    "new_year",
    "christmas",
    "labor_day",
    "independence_day",
    "revolution_day_memorial",
    "regional_games",
    "fifa_u_17_world_cup",
    "football_gold_cup",
    "beer_capital",
    "music_fest",
]

building_types = [
    'Other_Buildings',
    'Public',
    'University',
    'Data_Center',
    'Department_Store_and_Outlet',
    'Hospital',
    'Commercial',
    'Apartment',
    'Research_Institute',
    'Knowledge_Industry_Center',
    'Discount_Mart',
    'Hotel_and_Resort'
]


def read_df(train_csv, test_csv, building_info):
    columns = {
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
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

    train_df = pd.read_csv(train_csv, encoding="utf-8")
    test_df = pd.read_csv(test_csv, encoding="utf-8")
    building_df = pd.read_csv(building_info, encoding="utf-8")

    train_df = train_df.rename(columns=columns)
    train_df.drop("num_date_time", axis=1, inplace=True)
    train_df.drop("sunshine", axis=1, inplace=True)
    train_df.drop("solar_radiation", axis=1, inplace=True)

    test_df = test_df.rename(columns=columns)
    test_df.drop("num_date_time", axis=1, inplace=True)
    # test_df.drop("sunshine", axis=1, inplace=True)
    # test_df.drop("solar_radiation", axis=1, inplace=True)

    building_df = building_df.rename(columns=building_columns)
    building_df['building_type'] = building_df['building_type'].replace(translation_dict)

    train_df = pd.merge(train_df, building_df, on='building_number', how='left')
    test_df = pd.merge(test_df, building_df, on='building_number', how='left')

    train_df['date_time'] = pd.to_datetime(train_df['date_time'], format='%Y%m%d %H')
    train_df["time_idx"] = train_df["date_time"].dt.dayofyear * 24 + train_df['date_time'].dt.hour
    train_df["time_idx"] = train_df["time_idx"].astype("int")
    test_df['date_time'] = pd.to_datetime(test_df['date_time'], format='%Y%m%d %H')
    test_df["time_idx"] = test_df["date_time"].dt.dayofyear * 24 + test_df['date_time'].dt.hour
    test_df["time_idx"] = test_df["time_idx"].astype("int")

    train_df["building_number"] = train_df["building_number"].astype("str")
    test_df["building_number"] = test_df["building_number"].astype("str")

    # 설치했는지에대해 카테고리 처리?? 일단 0으로 변경
    train_df["solar_power_capacity"] = train_df.apply(
        lambda row: 0 if row["solar_power_capacity"] == "-" else row["solar_power_capacity"], axis=1)
    test_df["solar_power_capacity"] = train_df.apply(
        lambda row: 0 if row["solar_power_capacity"] == "-" else row["solar_power_capacity"], axis=1)

    train_df["ess_capacity"] = train_df.apply(
        lambda row: 0 if row["ess_capacity"] == "-" else row["ess_capacity"], axis=1)
    test_df["ess_capacity"] = train_df.apply(
        lambda row: 0 if row["ess_capacity"] == "-" else row["ess_capacity"], axis=1)

    train_df["pcs_capacity"] = train_df.apply(
        lambda row: 0 if row["pcs_capacity"] == "-" else row["pcs_capacity"], axis=1)
    test_df["pcs_capacity"] = train_df.apply(
        lambda row: 0 if row["pcs_capacity"] == "-" else row["pcs_capacity"], axis=1)

    return train_df, test_df, building_df


def prepare_data(train_df, test_df):
    # 추가 전처리
    # "special_days" 처리추가? 주말, 공휴일등
    # 추가한다면 create_dataSet의 time_varying_known_categoricals 추가

    # rainfall 결측치 처리
    train_df["rainfall"] = train_df["rainfall"].fillna(0.0)
    test_df["rainfall"] = train_df["rainfall"].fillna(0.0)

    # windspeed 결측치 처리
    train_df["windspeed"] = train_df["rainfall"].fillna(0.0)
    test_df["windspeed"] = train_df["rainfall"].fillna(0.0)
    # humidity 결측치 처리

    train_df["humidity"] = train_df["rainfall"].fillna(0.0)
    test_df["humidity"] = train_df["rainfall"].fillna(0.0)

    test_df["power_consumption"] = 0.0  # for test...

    return train_df, test_df


def create_dataSet(data, train_last_idx):
    # max_prediction_length = int(data["time_idx"].max() - train_last_idx)  # 24 * 7 Days...
    max_prediction_length = 24 * 7
    max_encoder_length = 24*7

    training = TimeSeriesDataSet(
        # data[lambda x: x.time_idx <= train_last_idx],
        data,
        time_idx="time_idx",
        target="power_consumption",
        group_ids=["building_number"],
        min_encoder_length=1,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["building_number", "building_type"],
        static_reals=["total_area", "cooling_area", "solar_power_capacity", "ess_capacity", "pcs_capacity"],
        time_varying_known_categoricals=[],
        # variable_groups={"building_type": building_types},
        # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "power_consumption",
            "temperature",
            "rainfall",
            "windspeed",
            "humidity",
        ],
        # target_normalizer=GroupNormalizer(
        #     groups=["building_number"], transformation="softplus"
        # ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    return training


def create_dataloader(trainSet, valid_set, test_df, num_workers=(0, 0)):
    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(trainSet, valid_set, predict=True, stop_randomization=True)
    test = TimeSeriesDataSet.from_dataset(trainSet, test_df, predict=True, stop_randomization=True)
    print(f"validation {len(validation)=}")
    print(f"test {len(test)=}")

    # create dataloaders for model
    batch_size = 128  # set this between 32 to 128
    train_dataloader = trainSet.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers[0])
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers[1])
    test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=num_workers[1])
    return train_dataloader, val_dataloader, test_dataloader


def find_optimal_learning_rate(training, train_dataloader, val_dataloader):
    trainer = pl.Trainer(
        accelerator=device,
        # clipping gradients is a hyperparameter and important to prevent divergance
        # of the gradient for recurrent neural networks
        gradient_clip_val=0.9,
    )
    # TODO:: 이거 ftf모델 파라미터 외부에서 가져와서 Tuner 돌리기
    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate=0.0015848931924611134,
        hidden_size=256,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=8,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=128,  # set to <= hidden_size
        loss=QuantileLoss(),
        optimizer="Ranger"
        # reduce learning rate if no improvement in validation loss after x epochs
        # reduce_on_plateau_patience=1000,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    from lightning.pytorch.tuner import Tuner

    res = Tuner(trainer).lr_find(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        max_lr=10.0,
        min_lr=1e-6,
    )

    print(f"suggested learning rate: {res.suggestion()}")
    # fig = res.plot(show=True, suggest=True)
    # fig.show()
    return res.suggestion()


def train(train_dataloader, val_dataloader, lr, max_epochs, train_set):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    #TODO:: Trainer 파라미터 수정, Validation 주기 등등 자세히 확인.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        val_check_interval=1.0,
        accelerator=device,
        enable_model_summary=True,
        gradient_clip_val=0.9,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    # TODO::  모델 파라미터 외부에서 가져오기.현재 하드 코딩.
    tft = TemporalFusionTransformer.from_dataset(
        train_set,
        learning_rate=lr,
        hidden_size=256,  # most important hyperparameter apart from learning rate
        # number of attention heads. Set to up to 4 for large datasets
        attention_head_size=8,
        dropout=0.1,  # between 0.1 and 0.3 are good values
        hidden_continuous_size=128,  # set to <= hidden_size
        loss=QuantileLoss(),
        optimizer="Ranger",
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size() / 1e3:.1f}k")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer


def optimize_hyperparameters(train_dataloader, val_dataloader):
    import pickle

    from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=100,
        max_epochs=5,
        timeout=3600 * 24,  # 1 Day
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(16, 512),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 8),
        learning_rate_range=(1e-5, 1.0),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=0.5),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=True,
        verbose=2# use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)


import datetime


def to_datetime(time_idx, base_time):
    dayofyear, hour = divmod(time_idx, 24)
    date = base_time + datetime.timedelta(days=dayofyear - 1, hours=hour)
    return date


def write_result(predictions, file_name):
    building_number = predictions.x["groups"].cpu()  # 100 x 1
    decoder_time_idx = predictions.x["decoder_time_idx"].cpu()  # 100 x 168

    output = predictions.output.cpu()  # 100 x 168
    base_time = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)

    lines = []

    lines.append(f"num_date_time,answer\n")
    for n in building_number[:, 0]:
        time_idx = decoder_time_idx[n]  # 168
        results = output[n]

        for idx, result in zip(time_idx, results):
            date = to_datetime(idx.item(), base_time)
            date_str = date.strftime("%Y%m%d %H")
            line = f"{n.item() + 1}_{date_str},{result.item()}\n"
            lines.append(line)

    with open(file_name, "w", encoding="utf-8") as f:
        f.writelines(lines)


if __name__ == '__main__':

    train_csv = "../Data/rawData/train.csv"
    test_csv = "../Data/rawData/test.csv"
    building_csv = "../Data/rawData/building_info.csv"
    train_num_worker, val_num_workers = (0, 0)
    opt_hyper_params = False
    do_train = False
    save_plot =False
    lr = 0.001
    max_epochs = 500

    train_df, test_df, building_df = read_df(train_csv, test_csv, building_csv)
    train_df, test_df = prepare_data(train_df, test_df)

    data_df = pd.concat([train_df, test_df], ignore_index=True)
    train_first_idx = train_df["time_idx"].min()
    train_last_idx = train_df["time_idx"].max()

    test_first_idx = test_df["time_idx"].min()
    test_last_idx = test_df["time_idx"].max()

    data_first_idx = data_df["time_idx"].min()
    data_last_idx = data_df["time_idx"].max()
    data_df.reset_index()
    print(f"{len(train_df)=} , {train_first_idx=},{train_last_idx=}")
    print(f"{len(test_df)=} , {test_first_idx=},{test_last_idx=}")
    print(f"{len(data_df)=} , {data_first_idx=},{data_last_idx=}")

    # print("Only TrainData")
    # train_set = create_dataSet(data_df, int(train_last_idx))
    # train_dataloader, val_dataloader, test_dataloader = create_dataloader(train_set,
    # data_df,
    # data_df,
    # num_workers = (
    #     train_num_worker, val_num_workers))

    print("Only TrainData")
    train_set = create_dataSet(train_df, int(train_last_idx))
    # 현재 test_라고 하지만 실재로는 val_dataloader
    train_dataloader, val_dataloader, test_dataloader = create_dataloader(train_set,
                                                                          train_df,
                                                                          data_df,
                                                                          num_workers=(
                                                                              train_num_worker, val_num_workers))

    if opt_hyper_params:
        optimize_hyperparameters(train_dataloader, val_dataloader)
        exit(1)

    if do_train:
        # lr = find_optimal_learning_rate(train_set, train_dataloader, val_dataloader)
        lr = 0.0015848931924611134
        trainer = train(train_dataloader, val_dataloader, lr, max_epochs, train_set)
        best_model_path = trainer.checkpoint_callback.best_model_path
        print(f"train {best_model_path=}")
    else:
        best_model_path = r"lightning_logs\\lightning_logs\\version_20\\checkpoints\\epoch=57-step=107474.ckpt"
        print(f"Saved {best_model_path=}")

    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    # print(f"{len(test_dataloader)=}")


    if save_plot:
        predictions = best_tft.predict(val_dataloader, return_x=True, return_y=True, mode="raw",
                                       trainer_kwargs=dict(accelerator=device))
        for i in range(100): # 100 : building numbers;
            fig = best_tft.plot_prediction(predictions.x, predictions.output, idx=i, add_loss_to_title=True)
            fig.savefig(f"plot_prediction_{i}.jpeg")
    #TODO::
    predictions = best_tft.predict(val_dataloader,  return_x=True, return_y=True,
                                   trainer_kwargs=dict(accelerator=device))
    smape_result = SMAPE()(predictions.output, predictions.y)
    mae_result = MAE()(predictions.output, predictions.y)
    print(predictions.output.shape)
    print(f"Validation(Train last 7Days) SMAPE : {smape_result}")
    print(f"Validation(Train last 7Days) MSE : {mae_result}")
    file_name = pathlib.Path(best_model_path).stem + "_val.csv"
    write_result(predictions, file_name)



    # todo!! Save Test Set Result
    predictions = best_tft.predict(test_dataloader, return_x=True, return_y=True,
                                   trainer_kwargs=dict(accelerator=device))
    file_name = pathlib.Path(best_model_path).stem + "_test.csv"
    print("write result : ", file_name)
    write_result(predictions, file_name)
    # result = SMAPE()(predictions.output, predictions.y)
    # print(f"Test (Test) SMAPE : {result}")
    # print(predictions.x)
