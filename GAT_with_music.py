import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Define the GAT model
class GATPredictor(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_heads):
        super(GATPredictor, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, num_classes, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
      
      
def nx_to_pyg_data(G):
    node_mapping = {node: i for i, node in enumerate(G.nodes)}
    edge_index = torch.tensor([(node_mapping[src], node_mapping[dst]) for src, dst in G.edges]).t().contiguous()
    x = torch.tensor([G.nodes[node]['music'] for node in G.nodes], dtype=torch.float)
    y = torch.tensor([G.nodes[node]['target'] for node in G.nodes], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)
 

# Convert the NetworkX graph to PyTorch Geometric data
train_data = nx_to_pyg_data(graph_train)
val_data = nx_to_pyg_data(graph_val)
test_data = nx_to_pyg_data(graph_test)


# Define the GAT model
num_features = len(G.nodes()[0]['music'])
hidden_dim = 128
num_classes = 18
num_heads = 8
model = GATPredictor(num_features, hidden_dim, num_classes, num_heads)

# Define the loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out, train_data.y)
    loss.backward()
    optimizer.step()

    # Perform validation
    model.eval()
    with torch.no_grad():
        val_out = model(val_data.x, val_data.edge_index)
        val_loss = criterion(val_out, val_data.y)
        
        _, predicted_labels = torch.max(val_out, dim=1)
        correct = (predicted_labels == val_data.y).sum().item()
        total = val_data.y.size(0)
        accuracy = correct / total * 100
    
    model.train()
    print(f"Epoch: {epoch+1}, Train Loss: {loss.item()}, Val Loss: {val_loss.item()}, Accuracy: {accuracy}")
 

# Get the test predictions
model.eval()
with torch.no_grad():
    test_out = model(test_data.x, test_data.edge_index)
    test_loss = criterion(test_out, test_data.y)
    _, predicted_labels = torch.max(test_out, dim=1)
    correct = (predicted_labels == test_data.y).sum().item()
    total = test_data.y.size(0)
    accuracy = correct / total * 100

print(f"Test Loss: {test_loss.item()}, Test Accuracy: {accuracy}")
      
