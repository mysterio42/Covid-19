import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from pylab import rcParams

def setup_params():
    sns.set(style='whitegrid', palette='muted', font_scale=1.2)

    HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

    sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

    rcParams['figure.figsize'] = 14, 10
    register_matplotlib_converters()

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

def prepare_data(file_name: str):
    df = pd.read_csv(file_name)
    df = df.iloc[:, 4:]
    assert df.isnull().sum().sum() == 0
    daily_cases = df.sum(axis=0)
    daily_cases.index = pd.to_datetime(daily_cases.index)
    # plot_data(daily_cases, 'Cumulative Daily Cases')
    print(daily_cases.shape)
    diff_daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
    # plot_data(diff_daily_cases, 'Difference of Daily Cases')
    return diff_daily_cases


def split_data(data, percent):
    total_data_size = data.shape[0]
    test_data_size = int(np.floor(total_data_size * percent / 100))
    train_data = data[:-test_data_size]
    test_data = data[-test_data_size:]
    return train_data, test_data


def scale_data(data, train_data, test_data=None):
    scaler = MinMaxScaler()
    scaler = scaler.fit(np.expand_dims(data, axis=1))
    train_data = scaler.transform(np.expand_dims(train_data, axis=1))
    if test_data is not None:
        test_data = scaler.transform(np.expand_dims(test_data, axis=1))
    return train_data, test_data, scaler


def create_sequences(data, seq_len: int):
    xs = []
    ys = []
    for i in range(len(data) - seq_len - 1):
        x = data[i:(i + seq_len)]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    return torch.from_numpy(np.array(xs)).float(), torch.from_numpy(np.array(ys)).float()
