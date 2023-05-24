import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json, sys, os

from model import VAE

# read file
with open(sys.argv[1], 'r') as f:
    inputs = json.loads(f.read())

num_digits = int(inputs['num_digits'])
latent_size = int(inputs['latent_size'])
model_path = str(inputs['model_path'])
data_dir = str(inputs['gen_data_dir'])

os.makedirs(data_dir, exist_ok=True)

# Set random seed for reproducibility
torch.manual_seed(42)


def generate_digits(vae, latent_size, num_digits=10, data_dir='./'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    vae.eval()

    encoded_digits = []
    image_paths = []

    with torch.no_grad():
        z = torch.randn(num_digits, latent_size).to(device)
        encoded_digits.append(z)
        generated_images = vae.decode(z).cpu()

    # Save generated images as PNG files
    for i in range(num_digits):
        digit_image = generated_images[i].squeeze().numpy()
        digit_image = (digit_image * 255).astype(np.uint8)
        image_path = os.path.join(data_dir, f"generated_digit_{i}.png")
        image_paths.append(image_path)
        plt.imsave(image_path, digit_image, cmap='gray')

        # Display the generated image
        plt.imshow(digit_image, cmap='gray')
        plt.axis('off')
        plt.show()

    # Create a DataFrame to store the encoded digits
    df = pd.DataFrame(encoded_digits[0].cpu().numpy(), columns=[f"in:{i}" for i in range(1, latent_size + 1)])

    # Add the image paths to the DataFrame
    df['img:digit'] = image_paths

    # Save the DataFrame as a CSV file
    csv_path = os.path.join(data_dir, 'design_explorer.csv')
    df.to_csv(csv_path, index=False)
    print(encoded_digits)

# Load the VAE model
def load_model(vae, model_path):
    vae.load_state_dict(torch.load(model_path))
    vae.eval()
    print(f"Model loaded from {model_path}")


# Load the VAE model
vae = VAE(latent_size = latent_size)
load_model(vae, model_path)

# Generate digits using the trained VAE
generate_digits(vae, latent_size, num_digits = num_digits, data_dir = data_dir)

