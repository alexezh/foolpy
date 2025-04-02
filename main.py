import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import model;
import data;

# code is based on
# https://github.com/pytorch/examples/blob/main/word_language_model/main.py

class Args:
    def __init__(self):
        # embedding size, 200 default
        self.emsize = 128;
        self.nhead = 8;
        # number of neurons per layer
        self.nhid = 64;
        # number of layers
        self.nlayers = 4;
        # small model
        self.dropout = 0.3;
        self.seed = 42;
        self.model = "Transformer"
        self.batch_size = 62;
        self.lr = 1
        self.epochs = 40
        # sequence length
        self.bptt = 64
        self.clip = 0.25
        self.log_interval = 10
        self.dry_run = False
        self.save = "model.pt"
        self.onnx_export = False
        self.temperature = 1.0
        self.cuda = False;
        self.mps = True;

args = Args();

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda.")
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    if not args.mps:
        print("WARNING: You have mps device, to enable macOS GPU run with --mps.")

use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")


corpus = data.Corpus(args.bptt)

eval_batch_size = 10;
train_data = DataLoader(corpus.train, args.batch_size)
val_data = DataLoader(corpus.valid, args.batch_size)
test_data = DataLoader(corpus.test, args.batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

#model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers).to(device)
model = model.RNNModel(ntokens, args.emsize, args.bptt, args.nhid, args.bptt).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # Ignore padding token
optimizer = optim.Adam(model.parameters(), lr=0.00005)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def complete(text: str):
    input = corpus.tokenize(text);
    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    input = torch.tensor(input).unsqueeze(0).to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary.idx2word)

    with torch.no_grad():
       for i in range(10):
                # data, targets = get_batch(source, 0)
#        if args.model == 'Transformer':
            output = model(input)

            next_word_logits = output[:, -1, :]  # Get last token predictions
            next_word_id = torch.argmax(F.softmax(next_word_logits, dim=-1), dim=-1)
       
            # word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            # word_idx = torch.multinomial(word_weights, 1)[0]
            # word_tensor = torch.Tensor([[next_word_id]]).long().to(device)
            
            # input = torch.cat(input, word_tensor).to(device)
            input = torch.cat([input, next_word_id.unsqueeze(0)], 1)

            w = corpus.dictionary.idx2word[next_word_id.item()]
            print(w);
            if w == '<eos>':
                break


#        else:
#            output, hidden = model(data, hidden)
#            hidden = repackage_hidden(hidden)

def train(epoch):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    
    batchIdx = 0;
    for batch, target in train_data:
        batch = batch.to(device)
        target = target.to(device)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()
        output = model(batch)
        output = output.view(-1, ntokens)
        target = target.view(-1)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batchIdx+=1

        if batchIdx % args.log_interval == 0 and batchIdx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batchIdx, len(train_data.dataset) // args.batch_size, 
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break

    return total_loss


def trainEpoc(): 
    # Loop over epochs.
    best_val_loss = None
    lr = args.lr

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            val_loss = train(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

runTrain = True
runTest = False

if runTrain:
    trainEpoc()

# Load the best saved model.
with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load(f))
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()

if runTest:
    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

else:
    complete("4 + 2 =>")



