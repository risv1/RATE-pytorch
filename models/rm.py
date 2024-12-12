import torch
import torch.nn as nn
from models.mrv import MemoryRetentionValve
from models.ct import CausalTransformer

class RATEModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, seq_length):
        super(RATEModel, self).__init__()
        self.returns_encoder = nn.Linear(1, embed_dim)
        self.observations_encoder = nn.Linear(seq_length, embed_dim)
        self.actions_encoder = nn.Linear(1, embed_dim)
        
        self.causal_transformer = CausalTransformer(embed_dim, num_heads, num_layers)
        self.memory_retention_valve = MemoryRetentionValve(embed_dim, num_heads)
    
    def forward(self, returns, observations, actions, prev_memory):
        # this is the encoding thingy
        rtg_encoded = self.returns_encoder(returns)
        obs_encoded = self.observations_encoder(observations)
        act_encoded = self.actions_encoder(actions)
        
        segment = rtg_encoded + obs_encoded + act_encoded
        
        # pass the segment through the transformer 🐱
        current_memory = self.causal_transformer(segment)
        
        # update the memory with the retention valve 🚰
        updated_memory = self.memory_retention_valve(prev_memory, current_memory)
        
        return updated_memory
    
    def reset_memory(self):
        self.memory_retention_valve.reset_cache()
