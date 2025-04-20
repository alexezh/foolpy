import torch
import torch.nn as nn
import torch.optim as optim

# maybe I need to go back to rnn, but again make it in a different way
# in markov, we say probability of X|Y. But in this case we want to
# fit result into pattern
# we say - 1 + 3 * 4 - I know from before to look at 3 * 4 as it was a rule
# so how do I encode the rule? We need a way to run sequence on CPU, and then
# optimize. So if we just run sequence rule: blah and then we train, we will get
# the result. Which means that we need network per-rule. Which we can try training for now
# 
# the rule is X, Y, Z

# Simple feedforward model
class CountModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.fc(x)

# Training loop
model = CountModel(input_size=10)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    x_batch, y_batch = generate_batch()
    output = model(x_batch)
    loss = criterion(output, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
