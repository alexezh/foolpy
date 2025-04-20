import time
import torch
import torch.nn as nn
import torch.optim as optim

from args import Args

# maybe I need to go back to rnn, but again make it in a different way
# in markov, we say probability of X|Y. But in this case we want to
# fit result into pattern
# we say - 1 + 3 * 4 - I know from before to look at 3 * 4 as it was a rule
# so how do I encode the rule? We need a way to run sequence on CPU, and then
# optimize. So if we just run sequence rule: blah and then we train, we will get
# the result. Which means that we need network per-rule. Which we can try training for now
#
# the logic can be thought as RNN, on short sequence with compression (keeping it short)
# we enter X + Y * Z - : system says mul. But later we learn the total cost can be lower
# so we apply different rule (scan blah). So we need to translate from dig + dig to num + num
# 
# the rule is X, Y, Z

# Simple feedforward model
class MemoizeModel(nn.Module):
    def __init__(self, args: Args, ntokens, embedding_weight):
        super().__init__()
        self.embedding = nn.Embedding(ntokens, args.emsize)
        self.embedding.load_state_dict(embedding_weight)
        self.embedding.weight.requires_grad = False

        self.fc1 = nn.Linear(args.emsize, args.nhid)  # First hidden layer
        self.fc2 = nn.Linear(args.nhid, args.nhid)  # Second hidden layer
        self.fc3 = nn.Linear(args.nhid, 1)  # Output layer (binary classification)

    def forward(self, x):
        embedded = self.embedding(x)
        
        # embedded_flat = embedded.view(embedded.size(0) * embedded.size(1), -1)  # [batch_size * seq_len, embedding_dim]

        hidden1 = torch.relu(self.fc1(embedded))  # Apply ReLU activation to first hidden layer
        hidden2 = torch.relu(self.fc2(hidden1))  # Apply ReLU to second hidden layer
        out = self.fc3(hidden2)  # Get the raw output
        return out;

criterion = None
optimizer = None
device = None
model = None

def initialize(args: Args, _device, ntokens, embedding_weight):
    global optimizer, criterion, device, model
    device = _device

    model = MemoizeModel(args, ntokens, embedding_weight).to(device)

    # Training loop
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

def train(train_data, epoch, args: Args):
    # Turn on training mode which enables dropout.
    model.train()
    start_time = time.time()
    
    batchIdx = 0;
    for src, tgt in train_data:
        src = src.to(device)
        tgt = tgt.float().to(device)
        # len = len.to(device)
        aux_tgt = aux_tgt.float().to(device)

        optimizer.zero_grad()
        probs, aux = model(src)  # [batch, seq_len]

        # Flatten outputs and targets for loss calculation
        probs = probs.view(-1)  # Flatten to shape [batch_size * seq_len]
        tgt = tgt.view(-1) 

        loss = criterion(probs, tgt)
        # flatten aux also
        # aux_loss = torch.sigmoid(aux)
        aux_loss = aux_criterion(aux.view(-1), aux_tgt.view(-1).float())
        # aux_loss = torch.sigmoid(aux_loss)

        # Learnable weight (scaled with exp to keep positive)
        # weight = torch.sigmoid(model.loss_weight)
        total_loss = loss + 0.7 * aux_loss

        total_loss.backward()
        optimizer.step()

        batch_loss += total_loss.item()
        
        batchIdx += 1

        if batchIdx % args.log_interval == 0 and batchIdx > 0:
            cur_loss = batch_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | loss {:5.2f} | {}'.format(
                epoch, batchIdx, len(train_data.dataset) // args.batch_size, 
                elapsed * 1000 / args.log_interval, cur_loss, str(aux_loss.item())))
            start_time = time.time()
            batch_loss = 0;

    return batch_loss

    x_batch, y_batch = generate_batch()
    output = model(x_batch)
    loss = criterion(output, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
