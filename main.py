from random import randint
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from args import Args
# import v2v;
import datacorpus
import lstm2;
import data
import rels;


# code is based on
# https://github.com/pytorch/examples/blob/main/word_language_model/main.py


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


corpus = datacorpus.Corpus(args.seq_length)

eval_batch_size = 10;
train_data = DataLoader(corpus.train, args.batch_size, shuffle=True, drop_last=True)
test_data = DataLoader(corpus.test, args.batch_size, shuffle=True, drop_last=True)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)

#model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers).to(device)
# model = model.RNNModel(ntokens, args.emsize, hidden_size=args.nhid, num_layers=args.nlayers, output_size=ntokens).to(device)
# model = lstm.Vector2VectorModel(input_dim=args.bptt, vocab_size=ntokens, embedding_dim=args.emsize, hidden_dim=args.bptt*2, output_dim=args.bptt).to(device)

def trainEpoc(): 
    # Loop over epochs.
    lr = args.lr

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            val_loss = lstm2.train(train_data, epoch, args)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                            val_loss))

        torch.save(model.state_dict(), args.save)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

runTrain = True
trainEmbedding = False
runTest = False

# relModel = rels.initialize();
embedding_weight = None
if trainEmbedding:
    embedding_weight = rels.train(args.emsize)
    torch.save(embedding_weight, args.relsFile)
else:
    with open(args.relsFile, 'rb') as f:
        embedding_weight = torch.load(f)

model = lstm2.initialize(args, device, ntokens, embedding_weight)

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

lstm2.complete("4 + 5 + 3", args, corpus)
lstm2.complete("4 + 5", args, corpus)
lstm2.complete("3 + 8 + 9", args, corpus)
lstm2.complete("5 + 3 + 8 + 9", args, corpus)
lstm2.complete("4 + 5 + 3 + 8 + 9", args, corpus)
lstm2.complete("a + 5 + 3", args, corpus)
lstm2.complete("3 + a + 3", args, corpus)



