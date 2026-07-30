import random
import numpy as np
import torch

def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set the seed globally
SEED = 10  #10, 20, 60
set_seed(SEED)

# Generator for DataLoader shuffle reproducibility (must match SEED)
g = torch.Generator()
g.manual_seed(SEED)

def seed_worker(worker_id):
    # Get the seed from the main process and apply it to the worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torchvision import models, transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score
from PIL import Image
import matplotlib.pyplot as plt
import copy

from tqdm.auto import tqdm

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
class Config:
    # Paths
    ROOT_DIR = '/kaggle/input/datasets/ayush1220/cifar10/cifar10'
    
    # Classes and Custom Mapping
    CLASSES = ['ship', 'dog', 'bird', 'airplane']
    # Mapping: ship=0, dog=1, bird=2, airplane=3
    TRAIN_SAMPLES = [2000, 1000, 600, 80] 
    
    # Autoencoder (DCGAN architecture from paper)
    LATENT_DIM = 128  # 600 for CIFAR-10 as per paper
    DIM_H = 64
    IMG_SIZE = 32
    
    # Training
    AE_EPOCHS = 100
    CLF_EPOCHS = 50
    BATCH_SIZE = 128
    LR = 0.0002
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = Config()

# ============================================================================
# 2. CUSTOM DATASET (No ImageFolder, direct folder reading)
# ============================================================================
class CustomCIFARDataset(Dataset):
    def __init__(self, root_dir, split='train', classes=None, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.split = split
        self.classes = classes
        self.transform = transform
        self.data = []
        self.targets = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, self.split, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: {cls_dir} not found!")
                continue
                
            img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Create imbalance for training set
            if split == 'train' and max_samples_per_class is not None:
                if len(img_files) > max_samples_per_class[idx]:
                    img_files = np.random.choice(img_files, max_samples_per_class[idx], replace=False)
            
            for img_file in img_files:
                self.data.append(os.path.join(cls_dir, img_file))
                self.targets.append(idx) # Custom mapping: 0, 1, 2, 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        target = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        return image, target

# Transforms
transform_all = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # -> [-1, 1]
])

# Update your dataset loading to use this:
train_dataset = CustomCIFARDataset(
    config.ROOT_DIR, 'train', config.CLASSES, transform_all, config.TRAIN_SAMPLES
)
test_dataset = CustomCIFARDataset(
    config.ROOT_DIR, 'test', config.CLASSES, transform_all
)



train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                          num_workers=2, worker_init_fn=seed_worker, generator=g)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=2, worker_init_fn=seed_worker)

print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

# ============================================================================
# 3. ENCODER & DECODER (DCGAN Architecture from Paper)
# ============================================================================
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
        )
        # 🔥 CHANGE 600 TO 128 HERE
        self.fc = nn.Linear(512 * 2 * 2, config.LATENT_DIM) 

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 🔥 CHANGE 600 TO 128 HERE
        self.fc = nn.Sequential(nn.Linear(config.LATENT_DIM, 512 * 2 * 2), nn.ReLU(True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1, bias=False), nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x).view(-1, 512, 2, 2)
        return self.deconv(x)

# ============================================================================
# 4. DEEPSMOTE TRAINING (With Exact Penalty Loss from Official Code)
# ============================================================================
def train_autoencoder(encoder, decoder, train_dataset, train_loader):
    enc_opt = optim.Adam(encoder.parameters(), lr=config.LR)
    dec_opt = optim.Adam(decoder.parameters(), lr=config.LR)
    criterion = nn.MSELoss()
    loss_history = []   # ADD

    print("Preloading images for penalty loss sampling...")
    all_imgs_tensor = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))]).to(config.DEVICE)

    class_indices = {}
    for cls in np.unique(train_dataset.targets):
        class_indices[cls] = np.where(np.array(train_dataset.targets) == cls)[0]

    encoder.train(); decoder.train()

    for epoch in range(config.AE_EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(config.DEVICE)
            z = encoder(images)
            x_recon = decoder(z)
            recon_loss = criterion(x_recon, images)

            target_cls = np.random.choice(list(class_indices.keys()))
            cls_idx_pool = class_indices[target_cls]
            n_samples = min(64, len(cls_idx_pool))
            sampled_idx = np.random.choice(cls_idx_pool, n_samples, replace=False)
            cls_imgs = all_imgs_tensor[sampled_idx]
            z_cls = encoder(cls_imgs)
            shifted_indices = torch.arange(1, n_samples).tolist() + [0]
            z_shifted = z_cls[shifted_indices]
            x_decoded_shifted = decoder(z_shifted)
            x_target_shifted = cls_imgs[shifted_indices]
            penalty_loss = criterion(x_decoded_shifted, x_target_shifted)

            loss = recon_loss + penalty_loss
            enc_opt.zero_grad(); dec_opt.zero_grad()
            loss.backward()
            enc_opt.step(); dec_opt.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        loss_history.append(epoch_loss)   # ADD
        if (epoch + 1) % 10 == 0:
            print(f"AE Epoch [{epoch+1}/{config.AE_EPOCHS}] Loss: {epoch_loss:.4f}")

    return loss_history   # ADD





def visualize_reconstruction(encoder, decoder, dataset, class_idx, class_name, config, n=8):
    encoder.eval(); decoder.eval()
    indices = [i for i, t in enumerate(dataset.targets) if t == class_idx][:n]
    real_imgs = torch.stack([dataset[i][0] for i in indices]).to(config.DEVICE)

    with torch.no_grad():
        z = encoder(real_imgs)
        recon_imgs = decoder(z).cpu()

    real_imgs = real_imgs.cpu()
    fig, axes = plt.subplots(2, n, figsize=(2*n, 5))
    for i in range(n):
        img_r = (real_imgs[i].permute(1,2,0).numpy() * 0.5) + 0.5
        axes[0, i].imshow(np.clip(img_r, 0, 1)); axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title(f"Real {class_name}", fontweight='bold', loc='left')

        img_d = (recon_imgs[i].permute(1,2,0).numpy() * 0.5) + 0.5
        axes[1, i].imshow(np.clip(img_d, 0, 1)); axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title("Reconstructed (no SMOTE)", fontweight='bold', loc='left')

    plt.tight_layout(); plt.show()



# ============================================================================
# 5. SMOTE IN LATENT SPACE
# ============================================================================
def generate_synthetic_for_all_classes(encoder, decoder, train_dataset, target_count):
    encoder.eval(); decoder.eval()
    
    all_synth_imgs = []
    all_synth_labels = []
    
    # Get unique classes
    unique_classes = np.unique(train_dataset.targets)
    
    for cls in unique_classes:
        # Get indices for this specific class
        cls_indices = [i for i, t in enumerate(train_dataset.targets) if t == cls]
        current_count = len(cls_indices)
        needed = target_count - current_count
        
        if needed <= 0:
            print(f"Class {cls}: Has {current_count} samples. Target is {target_count}. No generation needed.")
            continue
            
        print(f"Class {cls}: Generating {needed} synthetic samples...")
        
        # Create a temporary loader for this class
        cls_subset = Subset(train_dataset, cls_indices)
        cls_loader = DataLoader(cls_subset, batch_size=64, shuffle=False)
        
        # 1. Encode all real images of this class
        all_z = []
        with torch.no_grad():
            for imgs, _ in cls_loader:
                z = encoder(imgs.to(config.DEVICE))
                all_z.append(z.cpu().numpy())
                
        z_min = np.vstack(all_z)
        
        # 2. Apply SMOTE in latent space
        # Safety check: n_neighbors cannot be larger than the number of samples
        n_neighbors = min(6, current_count) 
        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        nn_model.fit(z_min)
        dist, ind = nn_model.kneighbors(z_min)
        
        synth_z = []
        for _ in range(needed):
            base = np.random.randint(0, current_count)
            neighbor = np.random.randint(1, n_neighbors)
            gap = np.random.uniform(0, 1.0)
            synth = z_min[base] + gap * (z_min[ind[base, neighbor]] - z_min[base])
            synth_z.append(synth)
            
        synth_z = torch.FloatTensor(np.array(synth_z)).to(config.DEVICE)
        
        # 3. Decode back to images
        synth_imgs = []
        for i in range(0, len(synth_z), 128):
            batch = synth_z[i:i+128]
            # 🔥 CRITICAL: .detach() prevents the RuntimeError we fixed earlier!
            synth_imgs.append(decoder(batch).detach().cpu()) 
            
        synth_imgs = torch.cat(synth_imgs, dim=0)
        
        all_synth_imgs.append(synth_imgs)
        all_synth_labels.append(torch.full((needed,), cls, dtype=torch.long))
        
    if len(all_synth_imgs) > 0:
        return torch.cat(all_synth_imgs, dim=0), torch.cat(all_synth_labels, dim=0)
    else:
        return torch.tensor([]), torch.tensor([])




import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def generate_synthetic_pcasmote(encoder, decoder, train_dataset, target_count, max_components=20):
    """
    PCA-SMOTE: Instead of SMOTE in the full latent space,
    project each class's latent vectors to their intrinsic low-dim subspace,
    do SMOTE there (where neighbors are meaningful), then project back.
    """
    encoder.eval(); decoder.eval()
    
    all_synth_imgs = []
    all_synth_labels = []
    
    unique_classes = np.unique(train_dataset.targets)
    
    print("\nPCA-SMOTE: Intrinsic dimensionality analysis")
    print("-" * 50)
    
    for cls in unique_classes:
        cls_indices = [i for i, t in enumerate(train_dataset.targets) if t == cls]
        current_count = len(cls_indices)
        needed = target_count - current_count
        
        if needed <= 0:
            print(f"Class {cls}: Has {current_count} samples. No generation needed.")
            continue
        
        # Encode all real images of this class
        cls_subset = Subset(train_dataset, cls_indices)
        cls_loader = DataLoader(cls_subset, batch_size=64, shuffle=False)
        all_z = []
        with torch.no_grad():
            for imgs, _ in cls_loader:
                all_z.append(encoder(imgs.to(config.DEVICE)).cpu().numpy())
        z_real = np.vstack(all_z)  # (current_count, latent_dim)
        
        # ── PCA: find intrinsic subspace ──────────────────────────────────────
        # n_components is at most n_samples-1 (PCA constraint) and at most max_components
        n_components = min(current_count - 1, max_components)
        pca = PCA(n_components=n_components)
        z_low = pca.fit_transform(z_real)  # (current_count, n_components)
        
        # Report explained variance
        cumvar = pca.explained_variance_ratio_.cumsum()
        dims_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        print(f"Class {cls} (n={current_count}): "
              f"PCA {n_components}d | "
              f"90% var in {dims_90} dims | "
              f"95% var in {dims_95} dims | "
              f"Generating {needed} samples...")
        
        # ── SMOTE in low-dim space (neighbors are now meaningful) ─────────────
        n_neighbors = min(6, current_count)
        nn_model = NearestNeighbors(n_neighbors=n_neighbors)
        nn_model.fit(z_low)
        dist, ind = nn_model.kneighbors(z_low)
        
        synth_z_low = []
        for _ in range(needed):
            base = np.random.randint(0, current_count)
            neighbor = np.random.randint(1, n_neighbors)
            gap = np.random.uniform(0, 1.0)
            synth = z_low[base] + gap * (z_low[ind[base, neighbor]] - z_low[base])
            synth_z_low.append(synth)
        
        # ── Project back to full latent space, then decode ────────────────────
        synth_z_high = pca.inverse_transform(np.array(synth_z_low))  # (needed, latent_dim)
        
        synth_z_tensor = torch.FloatTensor(synth_z_high).to(config.DEVICE)
        synth_imgs = []
        for i in range(0, len(synth_z_tensor), 128):
            synth_imgs.append(decoder(synth_z_tensor[i:i+128]).detach().cpu())
        synth_imgs = torch.cat(synth_imgs, dim=0)
        
        all_synth_imgs.append(synth_imgs)
        all_synth_labels.append(torch.full((needed,), cls, dtype=torch.long))
    
    print("-" * 50)
    if len(all_synth_imgs) > 0:
        return torch.cat(all_synth_imgs, dim=0), torch.cat(all_synth_labels, dim=0)
    else:
        return torch.tensor([]), torch.tensor([])


def visualize_generation(encoder, decoder, minority_loader, config, method_name="SMOTE"):
    encoder.eval(); decoder.eval()
    
    # Get 8 real minority images
    real_imgs, _ = next(iter(minority_loader))
    real_imgs = real_imgs[:8]
    
    # Encode them
    with torch.no_grad():
        z_real = encoder(real_imgs.to(config.DEVICE)).cpu().numpy()
        
    # Simple SMOTE interpolation between pairs
    z_synth = []
    for i in range(8):
        base = z_real[i]
        neighbor = z_real[(i + 1) % 8] # Just pair them up
        gap = np.random.uniform(0, 1.0)

        z_synth.append(base + gap * (neighbor - base))
        
    z_synth = torch.FloatTensor(np.array(z_synth)).to(config.DEVICE)
    
    # Decode synthetic
    with torch.no_grad():
        synth_imgs = decoder(z_synth).cpu()
        
    # Plot
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    
    for i in range(8):
        # Real
        img = real_imgs[i].permute(1, 2, 0).numpy() * std + mean
        axes[0, i].imshow(np.clip(img, 0, 1)); axes[0, i].axis('off')
        if i == 0: axes[0, i].set_title("Real Airplanes", fontweight='bold')
        
        # Synthetic
        img = synth_imgs[i].permute(1, 2, 0).numpy() * std + mean
        axes[1, i].imshow(np.clip(img, 0, 1)); axes[1, i].axis('off')
        if i == 0: axes[1, i].set_title(f"Synthetic ({method_name})", fontweight='bold')
        
    plt.suptitle(f"{method_name} Generation", fontsize=12)
    plt.tight_layout(); plt.show()




from sklearn.neighbors import NearestNeighbors

def check_neighbor_quality(encoder, dataset, class_idx, class_name, config):
    encoder.eval()
    indices = [i for i, t in enumerate(dataset.targets) if t == class_idx]
    imgs = torch.stack([dataset[i][0] for i in indices]).to(config.DEVICE)
    with torch.no_grad():
        z = encoder(imgs).cpu().numpy()

    n_neigh = min(6, len(z))
    nn_model = NearestNeighbors(n_neighbors=n_neigh)
    nn_model.fit(z)
    dist, ind = nn_model.kneighbors(z)
    neighbor_dists = dist[:, 1:]

    print(f"{class_name} (n={len(z)}):")
    print(f"  mean nearest-neighbor distance: {neighbor_dists[:,0].mean():.3f}")
    print(f"  mean latent vector norm: {np.linalg.norm(z, axis=1).mean():.3f}")
    print(f"  ratio: {neighbor_dists[:,0].mean() / np.linalg.norm(z, axis=1).mean():.3f}")
    return z, ind, dist



# ============================================================================
# 6. SMALL CNN CLASSIFIER (Fast, trained from scratch)
# ============================================================================
class SmallCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def get_classifier():
    # Returns a fresh SmallCNN to be trained from scratch
    return SmallCNN(num_classes=4)

def train_and_evaluate(model, train_loader, test_loader, name):
    model = model.to(config.DEVICE)
    
    # Adam is perfect for SmallCNN from scratch
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.CLF_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    for epoch in tqdm(range(config.CLF_EPOCHS)):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
                        
    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())
            
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    acc = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds, average='macro')
    
    class_accs = []
    for c in range(4):
        mask = all_true == c
        if mask.sum() > 0:
            class_accs.append((all_preds[mask] == c).sum() / mask.sum())
    acsa = np.mean(class_accs)
    gm = np.prod(class_accs) ** (1/len(class_accs))
    
    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ACSA: {acsa:.4f} | GM: {gm:.4f}")


    from sklearn.metrics import confusion_matrix
    print("Per-class accuracy:")
    for c, cname in enumerate(config.CLASSES):
        mask = all_true == c
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == c).sum() / mask.sum()
            print(f"  {cname}: {cls_acc:.4f} (n={mask.sum()})")

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion matrix (rows=true, cols=pred):", config.CLASSES)
    print(cm)

    
    return model


# ============================================================================
# 6b. WEIGHTED SYNTHETIC LOSS UTILITIES
# ============================================================================

def make_weighted_loader(original_imgs, original_labels, synth_imgs, synth_labels,
                         synth_weight=0.3, batch_size=100):
    """
    Build a DataLoader where real samples have loss weight=1.0
    and synthetic samples have loss weight=synth_weight.
    """
    real_weights = torch.ones(len(original_imgs))
    synth_weights_t = torch.full((len(synth_imgs),), float(synth_weight))

    all_imgs = torch.cat([original_imgs, synth_imgs])
    all_labels = torch.cat([original_labels, synth_labels])
    all_weights = torch.cat([real_weights, synth_weights_t])

    dataset = torch.utils.data.TensorDataset(all_imgs, all_labels, all_weights)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=2, worker_init_fn=seed_worker, generator=g)


def train_and_evaluate_weighted(model, train_loader, test_loader, name, synth_weight=0.3):
    """
    Same as train_and_evaluate but uses per-sample loss weighting.
    """
    print(f"  [Weighted training: real=1.0, synthetic={synth_weight}]")
    model = model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.CLF_EPOCHS)
    criterion = nn.CrossEntropyLoss(reduction='none')  # per-sample loss

    for epoch in tqdm(range(config.CLF_EPOCHS)):
        model.train()
        for batch in train_loader:
            imgs, labels, weights = batch
            imgs    = imgs.to(config.DEVICE)
            labels  = labels.to(config.DEVICE)
            weights = weights.to(config.DEVICE)   # (batch_size,)

            optimizer.zero_grad()
            per_sample_loss = criterion(model(imgs), labels)   # (batch_size,)
            loss = (per_sample_loss * weights).mean()          # weighted mean
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Evaluate
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(config.DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_true  = np.array(all_true)

    acc  = accuracy_score(all_true, all_preds)
    f1   = f1_score(all_true, all_preds, average='macro')

    class_accs = []
    for c in range(len(config.CLASSES)):
        mask = all_true == c
        if mask.sum() > 0:
            class_accs.append((all_preds[mask] == c).sum() / mask.sum())
    acsa = np.mean(class_accs)
    gm   = np.prod(class_accs) ** (1 / len(class_accs))

    print(f"\n--- {name} Results ---")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ACSA: {acsa:.4f} | GM: {gm:.4f}")

    from sklearn.metrics import confusion_matrix
    print("Per-class accuracy:")
    for c, cname in enumerate(config.CLASSES):
        mask = all_true == c
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == c).sum() / mask.sum()
            print(f"  {cname}: {cls_acc:.4f} (n={mask.sum()})")

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion matrix (rows=true, cols=pred):", config.CLASSES)
    print(cm)

    return model


# pyrefly: ignore [missing-import]
from torchmetrics.image.fid import FrechetInceptionDistance

# pyrefly: ignore [missing-import]
import lpips

# Clear cache before FID
import gc
torch.cuda.empty_cache()
gc.collect()

# Process in smaller batches to avoid OOM
def compute_fid_batched(real_imgs, synth_imgs, device, batch_size=64):
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Real features (smaller set)
    for i in range(0, len(real_imgs), batch_size):
        batch = ((real_imgs[i:i+batch_size] * 0.5) + 0.5).clamp(0,1).to(device)
        fid.update(batch, real=True)
    
    # Synthetic features (larger set)
    for i in range(0, len(synth_imgs), batch_size):
        batch = ((synth_imgs[i:i+batch_size] * 0.5) + 0.5).clamp(0,1).to(device)
        fid.update(batch, real=False)
    
    score = fid.compute().item()
    # Optional: reset to free memory
    fid.reset()
    return score

def compute_lpips_diversity(synth_imgs, device, n_pairs=50):
    loss_fn = lpips.LPIPS(net='alex').to(device)
    synth = synth_imgs.to(device)
    n = len(synth)
    dists = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, 2, replace=False)
        d = loss_fn(synth[i:i+1], synth[j:j+1])
        dists.append(d.item())
    return np.mean(dists)

# ============================================================================
# 7. MAIN EXECUTION PIPELINE
# ============================================================================
print("="*50)
print("PHASE 1: BASELINE (Imbalanced Data)")
print("="*50)
baseline_model = get_classifier()
train_and_evaluate(baseline_model, train_loader, test_loader, "Baseline (Imbalanced)")



print("\n" + "="*50)
print("PHASE 1.5: TRADITIONAL SMOTE (Balanced Data)")
print("="*50)

from sklearn.neighbors import NearestNeighbors

def apply_pixel_smote(train_dataset, target_count=2000):
    """Apply traditional SMOTE on flattened pixel space"""
    print("Applying traditional SMOTE on pixel space...")
    
    # Get all images and labels
    all_imgs = []
    all_labels = []
    for i in range(len(train_dataset)):
        img, label = train_dataset[i]
        all_imgs.append(img.numpy().flatten())  # Flatten to 1D
        all_labels.append(label)
    
    all_imgs = np.array(all_imgs)
    all_labels = np.array(all_labels)
    
    # Apply SMOTE for each underrepresented class
    synth_imgs_flat = []
    synth_labels = []
    
    for cls in np.unique(all_labels):
        cls_mask = all_labels == cls
        cls_imgs = all_imgs[cls_mask]
        current_count = len(cls_imgs)
        needed = target_count - current_count
        
        if needed <= 0:
            continue
            
        print(f"  Class {cls}: Generating {needed} samples via SMOTE...")
        
        # Fit k-NN
        n_neighbors = min(6, current_count)
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(cls_imgs)
        distances, indices = nn.kneighbors(cls_imgs)
        
        # Generate synthetic samples
        for _ in range(needed):
            base_idx = np.random.randint(0, current_count)
            neighbor_idx = np.random.randint(1, n_neighbors)
            gap = np.random.uniform(0, 1)
            
            synthetic = cls_imgs[base_idx] + gap * (
                cls_imgs[indices[base_idx, neighbor_idx]] - cls_imgs[base_idx]
            )
            synth_imgs_flat.append(synthetic)
            synth_labels.append(cls)
    
    # Reshape back to images
    if len(synth_imgs_flat) > 0:
        synth_imgs_flat = np.array(synth_imgs_flat)
        synth_imgs = synth_imgs_flat.reshape(-1, 3, 32, 32)  # CIFAR-10 shape
        synth_labels = np.array(synth_labels)
        return synth_imgs, synth_labels
    else:
        return np.array([]), np.array([])

# Apply traditional SMOTE
smote_imgs, smote_labels = apply_pixel_smote(train_dataset, target_count=2000)

# Create balanced dataset with SMOTE
original_imgs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
original_labels = torch.tensor(train_dataset.targets)

smote_imgs_tensor = torch.FloatTensor(smote_imgs)
smote_labels_tensor = torch.LongTensor(smote_labels)

all_imgs_smote = torch.cat([original_imgs, smote_imgs_tensor], dim=0)
all_labels_smote = torch.cat([original_labels, smote_labels_tensor], dim=0)

smote_dataset = torch.utils.data.TensorDataset(all_imgs_smote, all_labels_smote)
smote_loader = DataLoader(smote_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=seed_worker)

print(f"Traditional SMOTE - Total samples: {len(all_imgs_smote)}")

# Train and evaluate
smote_model = get_classifier()
train_and_evaluate(smote_model, smote_loader, test_loader, "Traditional SMOTE (Balanced)")




print("\n" + "="*50)
print("PHASE 2: DEEPSMOTE (Balancing Data)")
print("="*50)

# Train Autoencoder
encoder = Encoder().to(config.DEVICE)
decoder = Decoder().to(config.DEVICE)
loss_history = train_autoencoder(encoder, decoder, train_dataset, train_loader)   # capture return

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.xlabel("Epoch"); plt.ylabel("Total Loss (recon + penalty)")
plt.title("AE Training Curve")
plt.show()


# train_autoencoder(encoder, decoder, train_dataset, train_loader)

# class_idx=3 is airplane per your config.CLASSES mapping
visualize_reconstruction(encoder, decoder, train_dataset, class_idx=3, class_name="Airplane", config=config)




# 🔥 GENERATE FOR ALL CLASSES TO REACH 2000
print("\nGenerating synthetic images for ALL underrepresented classes...")
synth_imgs, synth_labels = generate_synthetic_for_all_classes(
    encoder, decoder, train_dataset, target_count=2000
)


# class 3 = airplane (minority, n=80), class 0 = ship (majority, n=2000) — direct comparison
_ = check_neighbor_quality(encoder, train_dataset, 3, "Airplane", config)
_ = check_neighbor_quality(encoder, train_dataset, 0, "Ship", config)


real_air_indices = [i for i, t in enumerate(train_dataset.targets) if t == 3]
real_air_imgs = torch.stack([train_dataset[i][0] for i in real_air_indices])

synth_air_mask = synth_labels == 3
synth_air_imgs = synth_imgs[synth_air_mask]

fid_score = compute_fid_batched(real_air_imgs, synth_air_imgs, config.DEVICE, batch_size=32)  # smaller batch if still OOM
diversity_score = compute_lpips_diversity(synth_air_imgs, config.DEVICE)
print(f"FID (real vs synthetic airplane): {fid_score:.2f}")
print(f"LPIPS diversity among synthetic airplanes: {diversity_score:.4f}")



print(f"Total synthetic images generated: {len(synth_imgs)}")

# Create balanced dataset
# Combine original train dataset with ALL synthetic images
original_imgs = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
original_labels = torch.tensor(train_dataset.targets)

all_imgs = torch.cat([original_imgs, synth_imgs])
all_labels = torch.cat([original_labels, synth_labels])

balanced_dataset = torch.utils.data.TensorDataset(all_imgs, all_labels)
balanced_loader = DataLoader(balanced_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=seed_worker )

print(f"Original dataset size: {len(original_imgs)}")
print(f"New balanced dataset size: {len(balanced_dataset)}") # This should now be exactly 8000!

# 🔥 FIX: Define minority_loader specifically for visualization (Class 3: Airplane)
minority_indices = [i for i, t in enumerate(train_dataset.targets) if t == 3]
minority_subset = Subset(train_dataset, minority_indices)
minority_loader = DataLoader(minority_subset, batch_size=8, shuffle=False, worker_init_fn=seed_worker)

# Now run the visualization
visualize_generation(encoder, decoder, minority_loader, config, method_name="DeepSMOTE")


print("\n" + "="*50)
print("PHASE 3: DEEPSMOTE CLASSIFIER (Balanced Data)")
print("="*50)
deepsmote_model = get_classifier()
train_and_evaluate(deepsmote_model, balanced_loader, test_loader, "DeepSMOTE (Balanced)")

print("\n✅ COMPLETE! Compare Phase 1 and Phase 3 metrics.")

# ============================================================================
# PHASE 4: PCA-SMOTE CLASSIFIER (Proposed Method)
# ============================================================================
print("\n" + "="*50)
print("PHASE 4: PCA-SMOTE (Proposed Method)")
print("="*50)
print("Reusing trained encoder/decoder from Phase 2.")
print("PCA reduces latent space to intrinsic dimensionality before SMOTE.")

# Generate with PCA-SMOTE
pca_synth_imgs, pca_synth_labels = generate_synthetic_pcasmote(
    encoder, decoder, train_dataset, target_count=2000, max_components=20
)

# FID and LPIPS for PCA-SMOTE minority class (class 3)
pca_min_mask = pca_synth_labels == 3
pca_min_imgs = pca_synth_imgs[pca_min_mask]

pca_fid_score = compute_fid_batched(real_air_imgs, pca_min_imgs, config.DEVICE, batch_size=32)
pca_diversity_score = compute_lpips_diversity(pca_min_imgs, config.DEVICE)
print(f"PCA-SMOTE FID (real vs synthetic Airplane): {pca_fid_score:.2f}")
print(f"PCA-SMOTE LPIPS diversity among synthetic Airplane: {pca_diversity_score:.4f}")

# Visualize PCA-SMOTE generated minority images vs real
visualize_generation(encoder, decoder, minority_loader, config, method_name="PCA-SMOTE")

# Build balanced dataset with PCA-SMOTE
original_imgs_pca = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
original_labels_pca = torch.tensor(train_dataset.targets)

all_imgs_pca = torch.cat([original_imgs_pca, pca_synth_imgs])
all_labels_pca = torch.cat([original_labels_pca, pca_synth_labels])

pca_balanced_dataset = torch.utils.data.TensorDataset(all_imgs_pca, all_labels_pca)
pca_balanced_loader = DataLoader(
    pca_balanced_dataset, batch_size=config.BATCH_SIZE,
    shuffle=True, num_workers=2, worker_init_fn=seed_worker, generator=g
)
print(f"PCA-SMOTE balanced dataset size: {len(pca_balanced_dataset)}")

# Train and evaluate
pca_model = get_classifier()
train_and_evaluate(pca_model, pca_balanced_loader, test_loader, "PCA-SMOTE (Proposed)")


# ============================================================================
# PHASE 3W: DEEPSMOTE + WEIGHTED LOSS (synthetic_weight ablation)
# ============================================================================
print("\n" + "="*50)
print("PHASE 3W: DEEPSMOTE + WEIGHTED SYNTHETIC LOSS")
print("="*50)
print("Same synthetic images as Phase 3 (DeepSMOTE, target=2000).")
print("Difference: synthetic samples downweighted in classifier loss.")

SYNTH_WEIGHT = 0.3  # tune: 0.1, 0.2, 0.3, 0.5
print(f"synth_weight = {SYNTH_WEIGHT}  (real=1.0, synthetic={SYNTH_WEIGHT})")

orig_imgs_base = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
orig_labels_base = torch.tensor(train_dataset.targets)

deep_weighted_loader = make_weighted_loader(
    orig_imgs_base, orig_labels_base,
    synth_imgs, synth_labels,
    synth_weight=SYNTH_WEIGHT,
    batch_size=config.BATCH_SIZE
)

deep_weighted_model = get_classifier()
train_and_evaluate_weighted(
    deep_weighted_model, deep_weighted_loader, test_loader,
    f"DeepSMOTE+Weighted (w={SYNTH_WEIGHT})",
    synth_weight=SYNTH_WEIGHT
)


# ============================================================================
# PHASE 4W: PCA-SMOTE + WEIGHTED LOSS (proposed method, full target=2000)
# ============================================================================
print("\n" + "="*50)
print("PHASE 4W: PCA-SMOTE + WEIGHTED SYNTHETIC LOSS")
print("="*50)
print("Same synthetic images as Phase 4 (PCA-SMOTE, target=2000).")
print("Difference: synthetic samples downweighted in classifier loss.")
print(f"synth_weight = {SYNTH_WEIGHT}  (real=1.0, synthetic={SYNTH_WEIGHT})")

pca_weighted_loader = make_weighted_loader(
    original_imgs_pca, original_labels_pca,
    pca_synth_imgs, pca_synth_labels,
    synth_weight=SYNTH_WEIGHT,
    batch_size=config.BATCH_SIZE
)

pca_weighted_model = get_classifier()
train_and_evaluate_weighted(
    pca_weighted_model, pca_weighted_loader, test_loader,
    f"PCA-SMOTE+Weighted (w={SYNTH_WEIGHT})",
    synth_weight=SYNTH_WEIGHT
)


print("\n" + "="*60)
print("FINAL COMPARISON SUMMARY")
print("="*60)
print("Compare all phases:")
print("  Phase 1   — Baseline (Imbalanced)")
print("  Phase 1.5 — Pixel SMOTE (Balanced, target=2000)")
print("  Phase 3   — DeepSMOTE (target=2000, unweighted)")
print("  Phase 4   — PCA-SMOTE (target=2000, unweighted)")
print(f"  Phase 3W  — DeepSMOTE (target=2000, synth_weight={SYNTH_WEIGHT})")
print(f"  Phase 4W  — PCA-SMOTE (target=2000, synth_weight={SYNTH_WEIGHT})  ← KEY")
print("Key question: Does Phase 4W beat both Phase 4 AND the Baseline?")
print("Key metric: Class-3 (Airplane) accuracy (minority class)")
