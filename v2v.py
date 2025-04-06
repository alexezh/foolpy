import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data import Corpus

class Vector2VectorModel(nn.Module):
    def __init__(self, input_dim, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()

        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x = self.embedding(x)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        output = self.layer4(x)  # No activation if it's a regression task
        return output

criterion = None
optimizer = None

def initialize(model):
    global optimizer;
    global criterion;
    
    criterion = nn.MSELoss()  # or CrossEntropyLoss if doing classification
    optimizer = optim.Adam(model.parameters(), lr=1)

def train(model, device, dataloader):
    # Train loop
    total_loss = 0

    for batch_inputs, batch_targets in dataloader:
        batch_inputs = batch_inputs.to(device).float()
        batch_targets = batch_targets.to(device).float()

        optimizer.zero_grad()
        outputs = model(batch_inputs)
        # outputs = torch.argmax(outputs, dim=-1)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss;

def complete(model, device, input, max_length, corpus: Corpus):
    ids = corpus.tokenize(input);
    ids = ids[:max_length] + [0] * (max_length - len(ids))

    input_tensor = torch.tensor(ids, dtype=torch.long)

    model.eval();

    with torch.no_grad():
        input_tensor = input_tensor.to(device).float()
        output = model(input_tensor)

        for val in torch.round(output).to(torch.int).flatten():
            print(corpus.dictionary.idx2word[val.item()]);
        #predicted_word_idx = torch.argmax(output[0, -1, :]).item()
