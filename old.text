





use_mps = args.mps and torch.backends.mps.is_available()
if args.cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.TransformerModel(200, emsize, nhead, nhid, nlayers, dropout).to(device)

# Define hyperparameters
input_dim = output_dim = 10000  # Vocabulary size
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 6
max_seq_length = 100

# Initialize model
model = model.CustomTransformer(input_dim, embed_dim, num_heads, ff_dim, num_layers, output_dim, max_seq_length, 0)
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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
src_input = src_input.to(device)

# Generate output sequence
predicted_output = model.predict(src_input, max_length=10, start_token=1, end_token=2)

print("Predicted Sequence:", predicted_output)