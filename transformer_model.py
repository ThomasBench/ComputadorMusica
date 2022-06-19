import torch 
import torch.nn as nn 
import math
from torch import Tensor
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads, 
        num_encoder_layers,
        num_decoder_layers, 
        forward_expansion, 
        dropout, 
        max_length, 
        device 
    ):
        super().__init__()
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size).to(device)
        self.src_position_embedding = PositionalEncoding( embedding_size,0.1,max_length).to(device)
        self.trg_position_embedding = PositionalEncoding( embedding_size,0.1,max_length).to(device)
        self.device = device
        self.transformer = nn.Transformer(
            embedding_size,
            num_heads, 
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout
        ).to(device)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size).to(device)
        self.dropout = nn.Dropout(dropout).to(device)
        self.softmax = nn.Softmax().to(device)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src: Tensor):
        shapes = src.shape
        mask = torch.zeros((1,shapes[0])).to(self.device)
        return mask
    def forward(self, src, trg ):

        # print(self.trg_position_embedding.get_device())
        embed_trg = self.trg_word_embedding(trg.to(torch.int64).to(self.device))

        
        embed_src = self.src_position_embedding(src) 
        embed_src = self.dropout( embed_src ).to(self.device)
        embed_trg = self.dropout( self.trg_position_embedding(embed_trg)).to(self.device)

        # print(embed_src.shape, embed_trg.shape)
        padding_mask = self.make_src_mask(src).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg.shape[0]).to(self.device)
        output = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask = padding_mask,
            tgt_mask = trg_mask
        )
        output = self.softmax(self.fc_out(output))
        return output