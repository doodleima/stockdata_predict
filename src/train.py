import os
import sys
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from pathlib import Path
sys.path.append(Path.cwd())

from model import SimpleLSTM , SimpleTransformer
from torch.utils.data import DataLoader, Subset
from dataloader import CustomStockDataset # load_and_preprocess_data


'''
mode = 학습 / 평가: train(학습 및 검증), test(평가)
epochs = epoch, lr = learning rate, weight_decay = decay value
'''
class stockAI:
    def __init__(self, mode='train', epochs=50, batches=64, lr=1e-04, weight_decay=1e-07):
        super(stockAI, self).__init__()
        self.mode = mode.lower()
        assert self.mode in ['train', 'test'], 'invalid mode name!'

        self.sequence_length = 10 # fixed
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.lr = lr
        self.epochs = epochs
        self.batches = batches
        self.weight_decay = weight_decay

        # self.model = SimpleTransformer(input_size=4, num_layers=20, output_size=1).to(self.device) # not implemented yet
        self.model = SimpleLSTM(input_size=4, num_layers=20, hidden_layer_size=100, output_size=1).to(self.device)
        self.loss = nn.MSELoss() # binary classification 
        self.optim = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


    def load_stock(self, inc_name):
        filename = os.path.join(Path.cwd(), 'data', self.mode, f'{inc_name}.csv')
        stock_dataset = CustomStockDataset(filename, self.sequence_length)

        return stock_dataset
    
    def run(self, inc_name):
        self.stock_dataset = self.load_stock(inc_name)
        self.best_score = np.inf

        if self.mode == 'train':
            train_size = int(0.8 * len(self.stock_dataset))            
            indices = torch.randperm(len(self.stock_dataset)).tolist()
            train_indices = indices[:train_size]
            valid_indices = indices[train_size:]

            train_dataset = Subset(self.stock_dataset, train_indices)
            valid_dataset = Subset(self.stock_dataset, valid_indices)

            train_loader = DataLoader(train_dataset, batch_size=self.batches, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=self.batches, shuffle=False)    # use all dataset for internal validation step: keeping consistency

            for epoch in tqdm(range(self.epochs)):
                train_pred, train_loss = self.train_one_epoch(train_loader)
                eval_pred, eval_loss = self.eval_one_epoch(valid_loader)

                tqdm.write(f'========================= [Epoch {epoch+1}] =========================')
                tqdm.write(f'Training Pred: {torch.mean(train_pred)}, Evaluation Pred: {torch.mean(eval_pred)}')
                tqdm.write(f'Training Loss: {train_loss}, Evaluation Loss: {eval_loss}')
                tqdm.write(f'=====================================================================')

                criteria = np.mean([train_loss, eval_loss])

                ### torch model save
                if criteria < self.best_score:
                    torch.save(self.model.state_dict(), os.path.join(Path.cwd(), "model", f"{inc_name}_best.pt"))
                    self.best_score = criteria


        else: # 'test'
            test_loader = DataLoader(self.stock_dataset, batch_size=self.batches)
            self.inference(test_loader)

    
    def train_one_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for sequences, labels in train_loader:
            sequences = sequences.to(self.device)
            labels = labels.unsqueeze(dim=1).to(self.device)

            self.optim.zero_grad()
            x_pred = self.model(sequences)

            loss = self.loss(x_pred, labels)
            loss.backward()
            self.optim.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        return torch.sigmoid(x_pred), avg_loss


    def eval_one_epoch(self, valid_loader):
        self.model.eval()
        total_eval_loss = 0

        with torch.no_grad():
            for sequences, labels in valid_loader:

                sequences = sequences.to(self.device)
                labels = labels.unsqueeze(dim=1).to(self.device)

                prediction = self.model(sequences)
                loss = self.loss(prediction, labels)
                total_eval_loss += loss.item()

        avg_loss = total_eval_loss / len(valid_loader)

        return torch.sigmoid(prediction), avg_loss


    def inference(self, test_loader, optimal_thres=0.5):
        self.model.eval()

        with torch.no_grad():
            for sequences, labels in test_loader:

                sequences = sequences.to(self.device)
                labels = labels.unsqueeze(dim=1).to(self.device)

                logit = self.model(sequences)
                conf_scores = torch.sigmoid(logit)
                print(conf_scores)

                ### 1 -> increased, 0 -> decreased
                ### due to data imbalance this result is not precised 
                for i in range(conf_scores.shape[0]):
                    pred_label = True if conf_scores[i].item() >= optimal_thres else False
                    gt_label = True if labels[i].item() == 1 else False                    
                    tqdm.write(f'[Predicted]: {pred_label}, [GT]: {gt_label}')
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, help="'train' or 'test'", required=True)
    parser.add_argument("-i", "--inc_name", type=str, help="incorporate name: 'LG전자', '네이버', ...", required=True)

    args = parser.parse_args()
    solver = stockAI(mode=args.mode)
    solver.run(inc_name=args.inc_name)