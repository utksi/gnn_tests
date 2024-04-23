import os
import argparse
import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F
import sys

def main():
    # Redirect stdout and stderr to a log file
    sys.stdout = open('train.log', 'w')
    sys.stderr = sys.stdout

    # Argument parsing for checkpoint handling
    parser = argparse.ArgumentParser(description="Train a GNN on the QM9 dataset.")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a checkpoint file to resume training.')
    parser.add_argument('--no-checkpoint', action='store_true',
                        help='Start training from scratch, ignoring any existing checkpoints.')
    args = parser.parse_args()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the QM9 dataset
    dataset = QM9(root="./QM9")
    train_dataset = dataset[:10000]
    test_dataset = dataset[10000:12000]

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Number of features per node
    num_features = train_dataset.num_node_features

    # Define the GNN model
    class GNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super(GNN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.out = torch.nn.Linear(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = global_mean_pool(x, batch)
            x = self.out(x)
            return x.view(-1)  # Flatten output

    model = GNN(num_features, 64, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()

    # Checkpoint functions
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(model, optimizer, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    # Determine how to initialize training
    start_epoch = 0
    if args.no_checkpoint:
        print("Starting training from scratch.")
    elif args.checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint)
        print(f"Resuming training from checkpoint: {args.checkpoint}")
    else:
        # Automatically look for the latest checkpoint
        checkpoint_files = sorted(os.listdir(checkpoint_dir))
        if checkpoint_files:
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            start_epoch = load_checkpoint(model, optimizer, latest_checkpoint)
            print(f"Automatically resuming from the latest checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoints found. Starting training from scratch.")

    # Training and evaluation loop
    test_loss = float('inf')
    epoch = start_epoch
    while test_loss > 0.005:
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = test(model, test_loader, criterion)
        epoch += 1
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

        if epoch % 100 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')
 
        if epoch > 1000000:
            print("Stopping early after 1,000,000 epochs.")
            break

    log_file.close()

if __name__ == "__main__":
    main()
