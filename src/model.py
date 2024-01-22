import torch
import numpy as np
import torch.nn as nn

### RNN (LSTM)
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, num_layers, hidden_layer_size, output_size):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x:torch.Tensor):
        x_out, _ = self.lstm(x)
        x_out_flatten = self.linear(x_out[:, -1, :])
        
        return x_out_flatten # no sigmoid applied


### Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)


    def forward(self, x:torch.Tensor):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    def __init__(self, input_size, num_layers, heads=4, dropout=0.2, output_size=1):
        super(SimpleTransformer, self).__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_size, dropout) # positional encoding
        encoder_layers = nn.TransformerEncoderLayer(d_model=input_size, nhead=heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(input_size, output_size)


    def forward(self, x:torch.Tensor):
        x = self.pos_encoder(x)
        x_out = self.transformer_encoder(x, self.src_mask)
        x_out = x_out.mean(dim=0)  # mean or sum: x_out.sum(dim=0)
        x_out_dec = self.decoder(x_out)

        return x_out_dec