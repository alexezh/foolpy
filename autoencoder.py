
import torch.nn as nn

def makeEncoder(input_dim, embedding_dim): 
  return nn.Sequential(
      nn.Linear(input_dim, 128),
      nn.ReLU(True),
      nn.Linear(128, embedding_dim)
  )

class ConceptAutoEncoder(nn.Module):
    
    def __init__(self, input_dim, embedding_dim):
        super(ConceptAutoEncoder, self).__init__()
        
        self.encoder = makeEncoder(input_dim, embedding_dim)
        
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
    
