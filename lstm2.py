import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import Args
from datacorpus import Corpus

class PositionSelector(nn.Module):
    def __init__(self, vocab_size, args: Args):
        super(PositionSelector, self).__init__()
#        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
#        self.classifier = nn.Linear(hidden_dim * 2, 1)  # 2 for bidirectional
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
    
#    def forward(self, x):  # x: [batch, seq_len, input_dim]
#        output, _ = self.lstm(x)
#        logits = self.classifier(output).squeeze(-1)  # [batch, seq_len]
#        probs = torch.sigmoid(logits)
#        return probs
    
criterion = None
optimizer = None
device = None
model = None

def initialize(args: Args, _device, ntokens):
    global optimizer, criterion, device, model
    device = _device

    model = PositionSelector(ntokens, args).to(device)

    pos_weight = torch.tensor([10.0]).to(device)  # weight ratio = (#zeros / #ones)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
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

