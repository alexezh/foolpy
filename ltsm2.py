import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import Args

class PositionSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PositionSelector, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, 1)  # 2 for bidirectional

    def forward(self, x):  # x: [batch, seq_len, input_dim]
        output, _ = self.lstm(x)
        logits = self.classifier(output).squeeze(-1)  # [batch, seq_len]
        probs = torch.sigmoid(logits)
        return probs
    
criterion = None
optimizer = None
device = None

def initialize(args: Args, _device):
    global optimizer, criterion, device
    device = _device

    model = PositionSelector(args.bptt, args.nhid).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    #criterion = nn.CrossEntropyLoss()

    return model


def complete(args, device, corpus, text: str):
    input = corpus.tokenize(text);
    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    input = torch.tensor([input]).to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary.idx2word)
    temperature = 0.7

    with torch.no_grad():
       for i in range(10):
                # data, targets = get_batch(source, 0)
#        if args.model == 'Transformer':
            output, hidden = model(input, hidden)

            predicted_word_idx = torch.argmax(output[0, -1, :]).item()
            # probs = F.softmax(output[:, -1, :] / temperature, dim=-1)
            # predicted_word_idx = torch.multinomial(probs, 1) 

            input = torch.tensor([[predicted_word_idx]]).to(device)
            w = corpus.dictionary.idx2word[predicted_word_idx]
            print(w);
            if w == '<eos>':
                break


def train(model, device, train_data, epoch, args):
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
            total_loss = 0
            start_time = time.time()

    return total_loss


#loss_weights = torch.ones(len(corpus.dictionary.word2idx)).to(device)
#loss_weights[corpus.dictionary.word2idx['<eos>']] = 5.0

#criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)  # Ignore padding token
#optimizer = optim.Adam(model.parameters(), lr=0.0005)

