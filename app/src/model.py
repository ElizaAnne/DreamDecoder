import torch
import torch.nn as nn
import math
from src.positional_encoding import PositionalEncoding

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers)
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, src, tgt):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        output = self.transformer(src, tgt)
        return self.fc_out(output)
