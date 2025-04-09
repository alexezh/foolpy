import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import Args

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers)

    def forward(self, src):
        # src shape: [seq_len, batch_size]
        embedded = self.embedding(src)  # [seq_len, batch_size, emb_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        # input shape: [1, batch_size]
        embedded = self.embedding(input)  # [1, batch_size, emb_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))  # [batch_size, output_dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [src_len, batch_size]
        # trg: [trg_len, batch_size]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)

        input = trg[0, :]  # First token (usually <sos>)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input.unsqueeze(0), hidden, cell)
            outputs[t] = output
            top1 = output.argmax(1)
            input = trg[t] if torch.rand(1).item() < teacher_forcing_ratio else top1

        return outputs

criterion = None
optimizer = None
device = None

def initialize(args: Args, _device):
    global optimizer, criterion, device
    device = _device

    encoder = Encoder(args.bptt, args.emsize, args.nhid).to(device)
    decoder = Decoder(args.bptt, args.emsize, args.nhid).to(device)
    model = Seq2Seq(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    return model


def complete(args, device, corpus, text: str):
    input = corpus.tokenize(text);
    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    input = torch.tensor([input]).to(device)

    hidden = torch.zeros(args.nlayers, 1, args.nhid).to(device)

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
        tgt = tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)       # skip <sos>
        tgt1 = tgt[1:].view(-1)

        loss = criterion(output, tgt1)
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

