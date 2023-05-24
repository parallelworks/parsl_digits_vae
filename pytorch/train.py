import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import json, sys

# read file
with open(sys.argv[1], 'r') as f:
    inputs = json.loads(f.read())

# Set random seed for reproducibility
torch.manual_seed(42)

from model import VAE

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    # KL divergence regularization
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training loop
def train_vae(vae, dataloader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.train()

    optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = vae(data)
            loss = vae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item()/len(data):.4f}")

        print(f"Epoch {epoch+1}/{num_epochs} | Avg. Loss: {total_loss / len(dataloader.dataset):.4f}")


# Save the VAE model
def save_model(vae, model_path):
    torch.save(vae.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# Load MNIST dataset
dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# Define latent size and create VAE instance
vae = VAE(latent_size=int(inputs['latent_size']))

# Train the VAE
train_vae(
    vae, 
    dataloader, 
    num_epochs = int(inputs['num_epochs']) , 
    learning_rate = float(inputs['learning_rate'])
)

# Save the trained VAE model
save_model(vae, inputs['model_path'])

