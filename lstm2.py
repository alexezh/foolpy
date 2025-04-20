import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import Args
from datacorpus import Corpus

def binary_concrete(logits, temperature=0.1):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-9) + 1e-9)
    y = torch.sigmoid((logits + gumbel) / temperature)
    return y

""" class ArithmeticEmbeddingLayer(nn.Module):
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
 """    
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

# class PositionalLinear(nn.Module):
#     def __init__(self, seq_len, in_features):
#         super().__init__()
#         self.position_embeddings = nn.Embedding(seq_len, in_features)

#     def forward(self, x):
#         batch_size, seq_len, in_features = x.size()
#         pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
#         pos_embed = self.position_embeddings(pos_ids)
#         return x + pos_embed
class PositionalLinear(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim

        # One linear layer per position
        self.weights = nn.Parameter(torch.randn(seq_len, input_dim, output_dim))
        self.biases = nn.Parameter(torch.zeros(seq_len, output_dim))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply the positional linear transformation
        out = torch.einsum('bsi,sio->bso', x, self.weights) + self.biases
        return out    

class PositionSelector(nn.Module):
    def __init__(self, args: Args, ntokens, embedding_weight):
        super(PositionSelector, self).__init__()
        
        self.embedding = nn.Embedding(ntokens, args.emsize)
        self.embedding.load_state_dict(embedding_weight)
        self.embedding.weight.requires_grad = False

        # Define the LSTM layer
        #self.lstm = nn.LSTM(input_size=args.emsize,  # Input size is 1 as we're not using embeddings
        #                    hidden_size=args.nhid,
        #                    num_layers=args.nlayers,
        #                    batch_first=True)
        
        # self.conv1 = nn.Conv1d(in_channels=args.emsize,
        #                        out_channels=args.nhid,
        #                        kernel_size=args.kernel_size,
        #                        padding=args.kernel_size // 2)  # to preserve seq length

        self.pos_linear = PositionalLinear(args.seq_length, args.emsize, args.nhid);

        # Fully connected layer to map to output
        # self.fc = nn.Linear(args.nhid, args.bptt)
        self.fc1 = nn.Linear(args.nhid, args.nhid)  # First hidden layer
        # self.fc2 = nn.Linear(args.emsize, args.nhid)  # Second hidden layer
        self.fc3 = nn.Linear(args.nhid, 1)  # Output layer (binary classification)

        # self.loss_weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32).to(device)) 
    
    def forward(self, x):
        embedded = self.embedding(x)
        
        # embedded_flat = embedded.view(embedded.size(0) * embedded.size(1), -1)  # [batch_size * seq_len, embedding_dim]
        # conv_input = embedded.permute(0, 2, 1) 
        # conv_out = F.relu(self.conv1(conv_input))        # [batch, hidden_dim, seq_len]
        # conv_out = conv_out.permute(0, 2, 1) 
        pos_out = self.pos_linear(embedded)

        hidden1 = torch.relu(self.fc1(pos_out))  # Apply ReLU activation to first hidden layer
        # hidden2 = torch.relu(self.fc2(hidden1))  # Apply ReLU to second hidden layer
        out = self.fc3(hidden1)  # Get the raw output
        
        # out = out.squeeze(-1);
        # out = out.view(x.size(0), x.size(1))
                  
        # x has shape [batch_size, seq_len]
        # We need to reshape it to [batch_size, seq_len, 1] for LSTM
        # x = x.unsqueeze(-1).float()  # Add a dimension and convert to float
        
        # lengths = lengths.view(-1);

        #sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
        #embedded = embedded[sorted_idx]  # Reorder embeddings based on sorted lengths

        #packed_input = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
                
        # Pass through LSTM
        # lstm_out, (hn, cn) = self.lstm(embedded)
        
        #lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
                
        # Use the last hidden state for prediction
        # out = self.fc(lstm_out[:, -1, :])  # Get the output of the last timestep

        # Apply sigmoid to produce values between 0 and 1 for position selection (binary)
        # out = torch.sigmoid(out)  # Output will be in the range [0, 1]
        # res = (torch.sigmoid(out.squeeze(-1)) > 0.5).int()   
        # aux_out = (res == 1).sum(dim=1).float().unsqueeze(1);
        # res = torch.sigmoid(out.squeeze(-1) * 10)
        res = binary_concrete(out.squeeze(-1), temperature=0.1)
        aux_out = res.sum(dim=1, keepdim=True)

        return out, aux_out

criterion = None
aux_criterion = None
optimizer = None
device = None
model = None

def initialize(args: Args, _device, ntokens, embedding_weight):
    global optimizer, criterion, device, model, aux_criterion
    device = _device

    model = PositionSelector(args, ntokens, embedding_weight).to(device)

    # pos_weight = torch.tensor([10.0]).to(device)  # weight ratio = (#zeros / #ones)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    aux_criterion = nn.MSELoss()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    #criterion = nn.CrossEntropyLoss()

    return model

def train(train_data, epoch, args: Args):
    # Turn on training mode which enables dropout.
    model.train()
    batch_loss = 0.
    start_time = time.time()
    
    batchIdx = 0;
    for src, tgt, srclen, aux_tgt in train_data:
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

def complete(text: str, args: Args, corpus: Corpus):
    input = corpus.tokenize(text);
    input = input[:args.seq_length] + [0] * (args.seq_length - len(input))

    #input = torch.tensor(input).type(torch.int64)
    #input = input.reshape(-1, 1).to(device)
    input = torch.tensor([input]).to(device)
    # input_len = torch.tensor([input_len]).to(device)

    # Turn on evaluation mode which disables dropout.
    model.eval()

    with torch.no_grad():
        output, output_len = model(input)
        print(output.shape)

        res = (output > 0.5).int().view(-1)   

        print(res);

#loss_weights = torch.ones(len(corpus.dictionary.word2idx)).to(device)
#loss_weights[corpus.dictionary.word2idx['<eos>']] = 5.0

#criterion = nn.CrossEntropyLoss(ignore_index=0, weight=loss_weights)  # Ignore padding token
#optimizer = optim.Adam(model.parameters(), lr=0.0005)

