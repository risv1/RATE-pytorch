import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from models.rm import RATEModel

def train_rate_model():
    embed_dim = 128
    num_heads = 8
    num_layers = 4
    seq_length = 10
    batch_size = 32
    num_epochs = 5
    num_segments = 3  # this is a simulation for trajectory and I made it 3 for simplicity

    model = RATEModel(embed_dim, num_heads, num_layers, seq_length)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    # this is dummy data lol
    returns = torch.rand(batch_size, num_segments, 1)
    observations = torch.rand(batch_size, num_segments, seq_length)
    actions = torch.rand(batch_size, num_segments, 1)
    prev_memory = torch.rand(batch_size, embed_dim)
    target = torch.rand(batch_size, embed_dim)

    memory_cache_over_time = []
    loss_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # reset the memory for each epoch so that the model can learn from scratch :D
        model.reset_memory()
        epoch_memory_states = []

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        for segment_idx in range(num_segments):
            output = model(
                returns[:, segment_idx],
                observations[:, segment_idx],
                actions[:, segment_idx],
                prev_memory
            )
            
            print(f"Segment {segment_idx + 1}/{num_segments}")
            print(f"Previous Memory (first 5 values): {prev_memory[0, :5].tolist()}")
            print(f"Updated Memory (first 5 values): {output[0, :5].tolist()}")

            # append the memory state for this segment 🗣️
            epoch_memory_states.append(output.clone().detach().cpu().numpy())
            
            # then update the previous memory with the current memory
            prev_memory = output.clone().detach()

        # this will be loss for final vs target memory
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        print(f"Epoch Loss: {loss.item()}")

        # store the memory states for this epoch 🥳
        memory_cache_over_time.append(epoch_memory_states)

    plot_results(loss_history, memory_cache_over_time)

def plot_results(loss_history, memory_cache_over_time):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    memory_cache = torch.tensor(memory_cache_over_time)
    for segment_idx in range(memory_cache.size(1)):
        segment_mem = memory_cache[:, segment_idx, 0, :5]
        plt.plot(segment_mem.numpy(), label=f"Segment {segment_idx + 1}")

    plt.xlabel("Epochs")
    plt.ylabel("Memory Embedding Values (first 5 dims)")
    plt.title("Memory Evolution over Epochs")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_rate_model()
