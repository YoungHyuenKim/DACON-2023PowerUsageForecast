import pandas as pd
import numpy as np
import random
import os
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

from model import LSTM
from preprocess import read_df, process_data
from dataset import create_data_loader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)  # Seed 고정

if __name__ == '__main__':
    input_size = 8  # feature의 개수
    hidden_size = 64
    num_layers = 2
    output_size = 1
    num_epochs = 5
    window_size = 48  # 예측에 사용될 시간 윈도우 크기
    batch_size = 64
    learning_rate = 0.001
    train_csv = "../Data/rawData/train.csv"
    test_csv = "../Data/rawData/test.csv"
    sample_submission_csv = "../Data/rawData/sample_submission.csv"
    now = datetime.datetime.now()

    run_time = now.strftime("%Y%m%d%H%M%S")
    output_submission_csv = f"./submission_{run_time}.csv"

    train_df, test_df, submission = read_df(train_csv, test_csv, sample_submission_csv)

    train_data, test_data, scaler = process_data(train_df, test_df, window_size=window_size)
    train_loader = create_data_loader(train_data, window_size, batch_size=batch_size, shuffle=True)
    test_loader = create_data_loader(test_df, window_size, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.unsqueeze(1).to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 300 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

    model.eval()
    test_predictions = []

    with torch.no_grad():
        for i in range(test_data.shape[0] - window_size):
            x = torch.Tensor(test_data[i:i + window_size, :]).to(device)
            new_x = model(x.view(1, window_size, -1))
            test_data[i + window_size, -1] = new_x  # 입력 업데이트
            test_predictions.append(new_x.detach().cpu().numpy().item())  # 예측 결과 저장

    predictions = scaler.inverse_transform(test_data)[window_size:, -1]  # 원래 scale로 복구
    submission['answer'] = predictions
    submission.to_csv(output_submission_csv, index=False)
