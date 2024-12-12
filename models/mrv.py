import torch
import torch.nn as nn

class MemoryRetentionValve(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MemoryRetentionValve, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.cache = None  # this is the hidden state cache storage thingy 😵
    
    def forward(self, prev_memory, current_memory):
        # if there is a cache, add the previous memory with the current memory
        if self.cache is not None:
            prev_memory = torch.cat([self.cache, prev_memory], dim=0)
        
        # combine both the memories with multi-head attention mechanism (I feel so typing saying that)
        query = current_memory
        key = torch.cat([prev_memory, current_memory], dim=0)
        value = torch.cat([prev_memory, current_memory], dim=0)
        
        # find attention and update memory
        attended_memory, _ = self.multihead_attention(query, key, value)
        updated_memory = self.layer_norm(attended_memory + current_memory)
        
        # update the cache with the current memory
        # detach the memory from the graph to prevent memory leaks 😇
        self.cache = updated_memory.clone().detach()
        
        return updated_memory
    
    def reset_cache(self):
        self.cache = None
