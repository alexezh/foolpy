import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import args
from device import device
import conceptdata
import autoencoder

criterion = None
optimizer = None
model = None

def initialize():
    global optimizer, criterion, model, aux_criterion

    model = autoencoder.ConceptAutoEncoder(conceptdata.string_enc_size, conceptdata.concept_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model

def trainWordEmbedding():
    initialize();
    data = []
    try:
        for epoch in range(1, 100):
            epoch_start_time = time.time()
            val_loss = trainWordEpoc(conceptdata.wordloader, epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                            val_loss))

        # now take embeddings and add them to trainset

        torch.save(model.state_dict(), "word")
        torch.save(model.encoder.state_dict(), "wordenc")

        testWord("hello")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

def loadWordEmbedding():
  initialize();

  with open(args.save, 'rb') as f:
    model.load_state_dict(torch.load("word"))

def loadWordEncoder():
  encoder = autoencoder.makeEncoder(conceptdata.string_enc_size, conceptdata.concept_size)
  with open("wordenc", 'rb') as f:
    encoder.load_state_dict(torch.load(f))

  return encoder

def trainWordEpoc(train_data, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    batch_loss = 0.
    start_time = time.time()
    # torch.autograd.set_detect_anomaly(True)
    batchIdx = 0;

    for src in train_data:
        src = src.to(device)

        optimizer.zero_grad()
        out, emb = model(src)  # [batch, seq_len]

        # Flatten outputs and targets for loss calculation
        out = out.view(-1)  # Flatten to shape [batch_size * seq_len]
        src = src.view(-1)

        loss = criterion(out, src)
        loss.backward()

        optimizer.step()

        batch_loss += loss.item()
        
        batchIdx += 1

        if batchIdx % args.log_interval == 0 and batchIdx > 0:
            print(model.fc3.weight.grad.norm())
            # print(model.pos_linear.weights.grad.norm())  # if using the einsum version

            # for name, param in model.named_parameters():
            #     print(param.requires_grad)
            #     print(name, param.grad.norm() if param.grad is not None else None)

            cur_loss = batch_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | {}'.format(
                epoch, batchIdx, len(train_data.dataset) // args.batch_size, 
                elapsed * 1000 / args.log_interval, cur_loss, "")) # str(aux_loss.item())
            start_time = time.time()
            batch_loss = 0;

    return batch_loss

def testWord(s):
    hot = conceptdata.string_to_onehot(s, conceptdata.max_string).view(-1)

    model.eval()

    with torch.no_grad():
        output, emb = model(hot)
        print(output.shape)

        probs = torch.sigmoid(output)
        probs = (probs > 0.5).int().view(-1)   

        print(probs);