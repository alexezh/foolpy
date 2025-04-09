import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import Args
from datacorpus import Corpus

class ArithmeticEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(ArithmeticEmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Manually initialize some relationships if desired
        # For example, we might want to initialize embeddings for certain words/entities.
        # You can initialize embeddings based on some prior knowledge about the relationships.
        # e.g., For entities "X" and "Y", their embeddings should be close to each other.
        # You can set specific values manually.
        
        # Example: Initializing embeddings for known relationships
        # If X is at index 0 and Y is at index 1, we can set their embeddings close
        self.embedding.weight.data[0] = torch.tensor([1.0, 0.5, 0.1])  # Embedding for X
        self.embedding.weight.data[1] = torch.tensor([1.1, 0.6, 0.1])  # Embedding for Y

    def forward(self, x):
        return self.embedding(x)
    
""" class PositionSelector(nn.Module):
    def __init__(self, vocab_size, args: Args):
        super(PositionSelector, self).__init__()
        self.embedding = nn.Embedding(vocab_size, args.emsize)
        self.dropout = nn.Dropout(0.5)
        self.bilstm = nn.LSTM(
            input_size = args.emsize, 
            hidden_size=args.nhid,
            num_layers=args.nlayers,
            batch_first=True, 
            bidirectional=True)
        self.fc = nn.Linear(args.nhid * 2, 1)  # output 1 logit per token

    def forward(self, x):
        # x: [batch_size, seq_len]
        embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        lstm_out, _ = self.bilstm(embed)  # [batch_size, seq_len, hidden_dim*2]

        lstm_out = self.dropout(lstm_out)

        logits = self.fc(lstm_out)  # [batch_size, seq_len, 1]
        logits = logits.squeeze(-1)  # [batch_size, seq_len]
        #probs = torch.sigmoid(logits)  # [batch_size, seq_len]
        #return probs
        return logits
    
 """    

import torch
import torch.nn as nn

class PositionSelector(nn.Module):
    def __init__(self, args: Args, ntokens, embedding_weight):
        super(PositionSelector, self).__init__()
        
        self.embedding = nn.Embedding(ntokens, args.emsize)
        self.embedding.load_state_dict(embedding_weight)
        self.embedding.weight.requires_grad = False

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=args.emsize,  # Input size is 1 as we're not using embeddings
                            hidden_size=args.nhid,
                            num_layers=args.nlayers,
                            batch_first=True)
        
        # Fully connected layer to map to output
        self.fc = nn.Linear(args.nhid, args.bptt)
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # x has shape [batch_size, seq_len]
        # We need to reshape it to [batch_size, seq_len, 1] for LSTM
        # x = x.unsqueeze(-1).float()  # Add a dimension and convert to float
        
        # Pass through LSTM
        lstm_out, (hn, cn) = self.lstm(embedded)
        
        # Use the last hidden state for prediction
        out = self.fc(lstm_out[:, -1, :])  # Get the output of the last timestep

        # Apply sigmoid to produce values between 0 and 1 for position selection (binary)
        # out = torch.sigmoid(out)  # Output will be in the range [0, 1]
                
        return out

criterion = None
optimizer = None
device = None
model = None

def initialize(args: Args, _device, ntokens, embedding_weight):
    global optimizer, criterion, device, model
    device = _device

    model = PositionSelector(args, ntokens, embedding_weight).to(device)

    pos_weight = torch.tensor([10.0]).to(device)  # weight ratio = (#zeros / #ones)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #criterion = nn.CrossEntropyLoss()

    return model


def complete(text: str, args: Args, corpus: Corpus):
    input = corpus.tokenize(text);
    input = input[:args.bptt] + [0] * (args.bptt - len(input))

    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    input = torch.tensor([input]).to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        output = model(input)
        print(output.shape)

        res = (output > 0.5).int().view(-1)   

        print(res);


def train(train_data, epoch, args: Args):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    
    batchIdx = 0;
    for src, tgt in train_data:
        src = src.to(device)
        tgt = tgt.float().to(device)

        optimizer.zero_grad()
        probs = model(src)  # [batch, seq_len]

        # Flatten outputs and targets for loss calculation
        probs = probs.view(-1)  # Flatten to shape [batch_size * seq_len]
        tgt = tgt.view(-1) 

        loss = criterion(probs, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        batchIdx += 1

        if batchIdx % args.log_interval == 0 and batchIdx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batchIdx, len(train_data.dataset) // args.batch_size, 
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            start_time = time.time()
            total_loss = 0;

    return total_loss


#loss_weights = torch.ones(len(corpus.dictionary.word2idx)).to(device)
#loss_weights[corpus.dictionary.word2idx['<eos>']] = 5.0

#criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)  # Ignore padding token
#optimizer = optim.Adam(model.parameters(), lr=0.0005)

