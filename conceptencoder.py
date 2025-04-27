import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import args
from device import device
import conceptdata


class ConceptAutoEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(ConceptAutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, embedding_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            # nn.Sigmoid()  # or remove if your input is not [0,1]
        )
        
    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    

criterion = None
optimizer = None
model = None

def initialize():
    global optimizer, criterion, model, aux_criterion

    model = ConceptAutoEncoder(conceptdata.concept_window * conceptdata.concept_size, conceptdata.concept_size).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model

def trainConceptEmbedding():
    initialize();
    data = []
    conceptdata.makeConcepts(data);
    try:
        for epoch in range(1, 100):
            epoch_start_time = time.time()
            val_loss = train(data, epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '.format(epoch, (time.time() - epoch_start_time),
                                            val_loss))

        torch.save(model.state_dict(), "concept")

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


def train(train_data, epoch):
    # Turn on training mode which enables dropout.
    model.train()
    batch_loss = 0.
    start_time = time.time()
    # torch.autograd.set_detect_anomaly(True)
    batchIdx = 0;

    for src in train_data:
        src = src.to(device)
        # tgt = tgt.float().to(device)
        # len = len.to(device)
        # aux_tgt = aux_tgt.float().to(device)

        optimizer.zero_grad()
        out, emb = model(src)  # [batch, seq_len]

        # Flatten outputs and targets for loss calculation
        out = out.view(-1)  # Flatten to shape [batch_size * seq_len]

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

def complete():
    # input = input[:args.seq_length] + [0] * (args.seq_length - len(input))

    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    # input = torch.tensor([input]).to(device)
    # input_len = torch.tensor([input_len]).to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        output = model(input)
        print(output.shape)

        probs = torch.sigmoid(output)
        probs = (probs > 0.5).int().view(-1)   

        print(probs);