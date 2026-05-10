import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from scipy.linalg import sqrtm
import numpy as np
from torch.utils.data import DataLoader, Subset
from models import Generator
import os

def get_inception_features(images, model, device):
    model.eval()
    features = []
    # Inception expects (N, 3, 299, 299)
    # MNIST is (N, 1, 28, 28)
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(device)
            batch = transform(batch)
            pred = model(batch)
            features.append(pred.cpu().numpy())
            
    return np.concatenate(features, axis=0)

def calculate_fid(real_features, fake_features):
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_inception_score(images, model, device, splits=1):
    model.eval()
    preds = []
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.Lambda(lambda x: x.repeat(1, 3, 1, 1) if x.size(1) == 1 else x),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        for i in range(0, len(images), 32):
            batch = images[i:i+32].to(device)
            batch = transform(batch)
            output = model(batch)
            preds.append(torch.nn.functional.softmax(output, dim=1).cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    
    scores = []
    for i in range(splits):
        part = preds[i * (len(preds) // splits): (i + 1) * (len(preds) // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
        
    return np.mean(scores), np.std(scores)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Inception model
    # We use the full model for IS and the features for FID
    # However, for FID we usually use the pool3 layer (2048 dims)
    # For simplicity and memory on CPU, we'll use the layer before the final FC
    inception_model = models.inception_v3(pretrained=True, transform_input=False)
    inception_model.fc = nn.Identity() # Remove final layer for FID
    inception_model.to(device)
    
    # For IS, we need the classification layer
    is_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    is_model.eval()
    
    # 1. Get Real Images (subset of 500 for speed)
    print("Loading real images...")
    dataset = datasets.MNIST(root="./data", train=True, transform=transforms.ToTensor(), download=True)
    subset_indices = list(range(500))
    real_images = torch.stack([dataset[i][0] for i in subset_indices])
    
    # 2. Generate Fake Images (500)
    print("Generating fake images...")
    netG = Generator(100).to(device)
    if os.path.exists("checkpoints/generator.pth"):
        netG.load_state_dict(torch.load("checkpoints/generator.pth", map_location=device))
    netG.eval()
    
    with torch.no_grad():
        noise = torch.randn(500, 100, device=device)
        fake_images = netG(noise).cpu()
        # Denormalize if they were Tanh outputted [-1, 1] -> [0, 1]
        fake_images = (fake_images + 1) / 2.0
        
    # 3. Calculate Features
    print("Extracting features (this may take a few minutes on CPU)...")
    real_features = get_inception_features(real_images, inception_model, device)
    fake_features = get_inception_features(fake_images, inception_model, device)
    
    # 4. Calculate FID
    print("Calculating FID...")
    fid_score = calculate_fid(real_features, fake_features)
    print(f"FID Score: {fid_score:.4f}")
    
    # 5. Calculate IS
    print("Calculating Inception Score...")
    is_mean, is_std = calculate_inception_score(fake_images, is_model, device)
    print(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}")
    
    with open("results/metrics.txt", "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
        f.write(f"Inception Score: {is_mean:.4f} +/- {is_std:.4f}\n")

if __name__ == "__main__":
    main()
