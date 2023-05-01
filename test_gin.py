import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GCNConv
from dgl.data.utils import load_graphs
from torch_geometric.utils import from_networkx
import dgl

class CustomDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)


    def get_num_features(self):
        sample_data = self.get(0)
        return sample_data.num_features


    @property
    def raw_file_names(self):
        # return [f for f in os.listdir(self.root) if f.endswith('.bin')]
        return [f for f in os.listdir(self.root) if f.endswith('.bin') and ("N" in f or "PB" in f or "UDH" in f or "DCIS" in f or "IC" in f)]

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        file_name = self.raw_file_names[idx]
        file_path = os.path.join(self.root, file_name)
        class_label = file_name.split('_')[2]

        # Load the DGL graph using DGL's load_graphs function
        dgl_graphs, _ = load_graphs(file_path)
        dgl_graph = dgl_graphs[0]

        # Used to check if our graph had edge features - it does not
        # if 'efeat' in dgl_graph.edata:
        #     edge_features = dgl_graph.edata['efeat']
        #     print("Graph has edge features with shape:", edge_features.shape)
        # else:
        #     print("Graph does not have edge features")

        # Get node features from DGL graph
        node_features = dgl_graph.ndata['feat']

        # Convert the DGL graph to a PyTorch Geometric Data object
        src, dst = dgl_graph.edges()
        edge_index = torch.stack((src, dst), dim=0).to(torch.long)
        data = Data(x=node_features, edge_index=edge_index)

        # Map the class label to an integer
        # class_mapping = {'N': 0, 'PB': 1, 'UDH': 2, 'FEA': 3, 'ADH': 4, 'DCIS': 5, 'IC': 6}
        # y = class_mapping[class_label]
        
        y = None
        if class_label in ["N", "PB", "UDH"]:
            y = 0
        elif class_label in ["DCIS", "IC"]:
            y = 1
        else:
            raise Exception("Filtered incorrectly for binary")
        
        data.y = torch.tensor([y], dtype=torch.long)
        return data
    
train_dataset = CustomDataset(root='/storage/home/hcocice1/gye31/cell_graph_dataset/train') # set path for all train graphs
val_dataset = CustomDataset(root='/storage/home/hcocice1/gye31/cell_graph_dataset/val')
test_dataset = CustomDataset(root='/storage/home/hcocice1/gye31/cell_graph_dataset/test')


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)
val_loader = DataLoader(val_dataset, batch_size=32)

print(f"Total dataset size: {len(train_dataset) + len(test_dataset) + len(val_dataset)}")
print(f"Number of features: {train_dataset.get_num_features()}")
print(f"Number of classes: {train_dataset.num_classes}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_max_pool, global_add_pool, global_mean_pool

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=256):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            Sequential(Linear(num_features, hidden_size), BatchNorm1d(hidden_size),
                       ReLU(), Dropout(p=0.5),
                       Linear(hidden_size, hidden_size), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_size, hidden_size), ReLU(), BatchNorm1d(hidden_size), Dropout(p=0.5),
                       Linear(hidden_size, hidden_size), ReLU()))
        # self.conv3 = GINConv(
        #     Sequential(Linear(hidden_size, hidden_size), ReLU(), BatchNorm1d(hidden_size),
        #                Linear(hidden_size, hidden_size), ReLU()))
        # self.conv4 = GINConv(
        #     Sequential(Linear(hidden_size, hidden_size), ReLU(), BatchNorm1d(hidden_size),
        #                Linear(hidden_size, hidden_size), ReLU()))
        # self.conv5 = GINConv(
        #     Sequential(Linear(hidden_size, hidden_size), ReLU(), BatchNorm1d(hidden_size),
        #                Linear(hidden_size, hidden_size), ReLU()))
        self.lin1 = Linear(hidden_size * 2, hidden_size // 2)
        self.lin2 = Linear(hidden_size // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        # x = self.conv3(x, edge_index)
        # x = self.conv4(x, edge_index)
        # x = self.conv5(x, edge_index)
        h1 = global_add_pool(h1, batch)
        h2 = global_add_pool(h2, batch)
        x = torch.cat((h1, h2), dim=1)
        # # Graph-level readout
        # x = global_mean_pool(x, batch)

        # Classifier
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


# Initialize the GIN
hidden_size = 1024
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(train_dataset.num_features, train_dataset.num_classes, hidden_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
best_val_accuracy = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, data.y)
        loss.backward()
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs
        optimizer.step()
        total_loss += loss.item()
    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            loss = F.cross_entropy(output, data.y)
            val_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(data.y).sum().item()
            total += data.num_graphs
    val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    
    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_gin_model.pth")
    
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train Accuracy: {train_accuracy:.4f}")

# Testing loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(data.y).sum().item()
        total += data.num_graphs

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")