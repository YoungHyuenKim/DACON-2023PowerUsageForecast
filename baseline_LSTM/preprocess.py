import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def read_df(train_csv, test_csv, submission_csv):
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    sample_submission = pd.read_csv(submission_csv)

    return train_df, test_df, sample_submission


def train_data_preprocessing(train_df):
    remove_field = ['일조(hr)', '일사(MJ/m2)']

    train_df = train_df.drop(remove_field, axis=1)
    train_df['강수량(mm)'].fillna(0.0, inplace=True)
    train_df['풍속(m/s)'].fillna(round(train_df['풍속(m/s)'].mean(), 2), inplace=True)
    train_df['습도(%)'].fillna(round(train_df['습도(%)'].mean(), 2), inplace=True)
    train_df['month'] = train_df['일시'].apply(lambda x: float(x[4:6]))
    train_df['day'] = train_df['일시'].apply(lambda x: float(x[6:8]))
    train_df['time'] = train_df['일시'].apply(lambda x: float(x[9:11]))
    train_df = train_df[
        train_df.columns[:7].to_list() + train_df.columns[8:].to_list() + train_df.columns[7:8].to_list()]
    return train_df


def test_data_preprocessing(test_df):
    # 실수형 데이터로 변환
    test_df['습도(%)'] = test_df['습도(%)'].astype('float64')

    # 날짜 데이터 추가
    test_df['month'] = test_df['일시'].apply(lambda x: float(x[4:6]))
    test_df['day'] = test_df['일시'].apply(lambda x: float(x[6:8]))
    test_df['time'] = test_df['일시'].apply(lambda x: float(x[9:11]))
    final_df = pd.concat(
        (test_df.drop(['num_date_time', '건물번호', '일시', ], axis=1), pd.DataFrame(np.zeros(test_df.shape[0]))), axis=1)
    final_df = final_df.rename({0: '전력소비량(kWh)'}, axis=1)
    return final_df


def process_data(train_df, test_df, window_size):
    train_df = train_data_preprocessing(train_df)
    # last_train_data = train_df.drop(['num_date_time', '건물번호', '일시', ], axis=1).loc[204000 - 24:, :]
    last_train_data = train_df.drop(['num_date_time', '건물번호', '일시', ], axis=1).loc[204000 - window_size:, :]
    test_df = test_data_preprocessing(test_df)
    test_df = pd.concat((last_train_data, test_df)).reset_index(drop=True)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df.drop(['num_date_time', '건물번호', '일시'], axis=1).values)
    test_data = scaler.transform(test_df.values)  # train과 동일하게 scaling
    return train_data, test_data, scaler
