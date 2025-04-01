import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
from torch.utils.data import Dataset, DataLoader

import model;
import data;

# code is based on
# https://github.com/pytorch/examples/blob/main/word_language_model/main.py

class Args:
    def __init__(self):
        # embedding size, 200 default
        self.emsize = 64;
        self.nhead = 1;
        # number of neurons per layer
        self.nhid = 64;
        # number of layers
        self.nlayers = 2;
        # small model
        self.dropout = 0.3;
        self.seed = 42;
        self.model = "Transformer"
        self.batch_size = 512;
        self.lr = 10
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
model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)

criterion = nn.NLLLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for batch, target in data_source:
            batch = batch.to(device)
            target = target.to(device)
            output = model(batch)
            output = output.view(-1, ntokens)
            target = target.view(-1)
            total_loss += len(batch) * criterion(output, target).item()
    return total_loss / (len(data_source.dataset) - 1)

def complete(text: str):
    input = corpus.tokenize(text);
    input = input.reshape(-1, 1).contiguous().to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(corpus.dictionary.idx2word)

    with torch.no_grad():
       for i in range(10):
                # data, targets = get_batch(source, 0)
#        if args.model == 'Transformer':
            output = model(input)
            output = output.view(-1, ntokens)

            word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            word_tensor = torch.Tensor([[word_idx]]).long().to(device)
            
            input = torch.cat([input, word_tensor], 0)

            w = corpus.dictionary.idx2word[word_idx]
            print(w);
            if w == '<eos>':
                break


#        else:
#            output, hidden = model(data, hidden)
#            hidden = repackage_hidden(hidden)

def train(epoch, lr):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)
    
    batchIdx = 0;
    for batch, target in train_data:
        batch = batch.to(device)
        target = target.to(device)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        output = model(batch)
        output = output.view(-1, ntokens)
        target = target.view(-1)

        loss = criterion(output, target)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        batchIdx+=1

        if batchIdx % args.log_interval == 0 and batchIdx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batchIdx, len(train_data.dataset) // args.batch_size, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run:
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)



def trainEpoc(): 
    # Loop over epochs.
    best_val_loss = None
    lr = args.lr

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(epoch, lr)
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            val_loss, math.exp(val_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

runTrain = False
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

    #if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
    #    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
else:
    complete("4 + 2 =>")



