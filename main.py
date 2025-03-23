import torch
import torch.nn as nn
import torch.optim as optim

class CustomTransformer(nn.Module):
    class CustomTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, max_seq_length):
        super(CustomTransformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_seq_length, embed_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, output_dim)

    def forward(self, src, tgt):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        output = self.fc_out(output)
        return output

    def predict(self, src, max_length=20, start_token=1):
        """ Generate predictions given input sequence """
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        memory = self.encoder(src)

        # Start with the start token
        tgt = torch.full((src.size(0), 1), start_token, dtype=torch.long, device=src.device)

        for _ in range(max_length):
            tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
            output = self.decoder(tgt_emb, memory)
            output = self.fc_out(output)

            # Get the last token prediction
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(-1)
            tgt = torch.cat((tgt, next_token), dim=1)

        return tgt
    
# Define hyperparameters
input_dim = output_dim = 10000  # Vocabulary size
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
max_seq_length = 100

# Initialize model
model = CustomTransformer(input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, max_seq_length)
print(model)


# Dummy data
src = torch.randint(0, 10000, (10, 20))  # (batch_size, sequence_length)
tgt = torch.randint(0, 10000, (10, 20))

# Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
src, tgt = src.to(device), tgt.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training step
for epoch in range(5):  # Small number of epochs for testing
    optimizer.zero_grad()
    output = model(src, tgt)
    loss = criterion(output.view(-1, output_dim), tgt.view(-1))  # Reshape for loss calculation
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# predict
# Define dummy input (batch_size=1, sequence_length=10)
src_input = torch.randint(0, 10000, (1, 10))

# Move model and input to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
src_input = src_input.to(device)

# Generate output sequence
predicted_output = model.predict(src_input, max_length=15, start_token=2)

print("Predicted Sequence:", predicted_output)