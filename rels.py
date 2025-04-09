import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import data
import datacorpus

class RelsAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

        # Reconstruct the original tokens
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # Flatten across sequence
        batch_size, seq_len, embed_dim = embedded.shape
        embedded_flat = embedded.view(-1, embed_dim)

        encoded = self.encoder(embedded_flat)       # [batch_size * seq_len, hidden_dim]
        decoded = self.decoder(encoded)             # [batch_size * seq_len, embed_dim]
        logits = self.output_layer(decoded)         # [batch_size * seq_len, vocab_size]

        return logits.view(batch_size, seq_len, -1)  # [batch_size, seq_len, vocab_size]

eval_batch_size = 10;
rels = [];
data.makeRels(rels);
train_set = datacorpus.TokenizedDataset.makeRel(rels, datacorpus.dictionary.word2idx, 1)
train_data = DataLoader(train_set, 1, shuffle=True, drop_last=True)

def train(embed_dim):

  ntokens = len(datacorpus.dictionary.idx2word)
  # model = RelsAutoencoder(ntokens, embed_dim=embed_dim, hidden_dim=1)

  embedding = nn.Embedding(ntokens, embed_dim)
  optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-4)
  # loss_fn = nn.CrossEntropyLoss()  # Use token IDs as targets
  loss_fn = nn.MSELoss();

  for epoch in range(40):
      for src, tgt in train_data:
          # inputs = src   # assume [batch_size, seq_len] of token IDs

          emb_src = embedding(src)
          emb_tgt = embedding(tgt)
          # outputs = model(inputs)  # [batch, seq_len, vocab_size]

          # loss = loss_fn(outputs.view(-1, ntokens), tgt.view(-1))
          loss = loss_fn(emb_src, emb_tgt);

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

      print(f"Epoch {epoch} Loss: {loss.item():.4f}")

  embedding_weights = embedding.state_dict()
  return embedding_weights