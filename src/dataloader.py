import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
    

class CustomStockDataset(Dataset):
    def __init__(self, filename, sequence_length):
        self.sequence_length = sequence_length
        self.data, self.labels = self.preprocessing(filename)

    ### data & label generation
    def preprocessing(self, filename):
        data = pd.read_csv(filename)

        data['Label'] = np.nan
        
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        weekly_avg = data['Close'].resample('W').mean() # weekly average value

        weekly_change = weekly_avg.pct_change()
        labels = np.where(weekly_change > 0, 1, 0) 
        labels = np.roll(labels, -1)[:-1]  # 레이블을 하나씩 shift하고 마지막 주 제거

        ### preprocessing: weights like 'ensemble'
        data = data[['Open', 'High', 'Low', 'Close']]
        data['Open'] *= 0.2
        data['High'] *= 0.1
        data['Low'] *= 0.1
        data['Close'] *= 0.6

        ### applied normalization: intensity 0 to 1
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        for i, date in enumerate(weekly_avg.index):
            if i < len(labels): data.loc[data.index.week == date.week, 'Label'] = labels[i]

        data.dropna(inplace=True)

        return scaled_data, data['Label'].values


    def __len__(self):
        return len(self.data) - self.sequence_length


    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]

        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)