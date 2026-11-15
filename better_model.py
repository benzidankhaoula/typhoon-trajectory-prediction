
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =========================
# CONFIGURATION
# =========================
CSV_PATH = "data/ibtracs_wp.csv"
SEQ_LEN = 12          
PRED_LEN = 4          
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# DATASET 
# =========================
class SimpleTyphoonDataset(Dataset):
    def __init__(self, csv_path, seq_len=12, pred_len=4):
        print("\nLoading data...")

        df = pd.read_csv(csv_path, usecols=['SID', 'LAT', 'LON'])
        
        df['LAT'] = pd.to_numeric(df['LAT'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LON'], errors='coerce')
        df = df.dropna()
        df = df[(df['LAT'] >= 0) & (df['LAT'] <= 50) & 
                (df['LON'] >= 105) & (df['LON'] <= 165)]
        
        print(f"After filtering: {len(df)} rows")
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.samples = []
        
        for sid, group in df.groupby('SID'):
            coords = group[['LAT', 'LON']].values
            
            if len(coords) < seq_len + pred_len:
                continue
            
            for i in range(len(coords) - seq_len - pred_len + 1):
                x = coords[i:i+seq_len]
                y = coords[i+seq_len:i+seq_len+pred_len]
                self.samples.append((x, y))
        
        print(f"Created {len(self.samples)} samples")
        
        all_lats = []
        all_lons = []
        for x, _ in self.samples:
            all_lats.extend(x[:, 0])
            all_lons.extend(x[:, 1])
        
        self.lat_min, self.lat_max = min(all_lats), max(all_lats)
        self.lon_min, self.lon_max = min(all_lons), max(all_lons)
        
        print(f"\nNormalization ranges:")
        print(f"  LAT: [{self.lat_min:.1f}, {self.lat_max:.1f}]")
        print(f"  LON: [{self.lon_min:.1f}, {self.lon_max:.1f}]")
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        x, y = self.samples[idx]
        
        # Min-max normalization 
        x_norm = np.zeros_like(x)
        y_norm = np.zeros_like(y)
        
        x_norm[:, 0] = (x[:, 0] - self.lat_min) / (self.lat_max - self.lat_min)
        x_norm[:, 1] = (x[:, 1] - self.lon_min) / (self.lon_max - self.lon_min)
        
        y_norm[:, 0] = (y[:, 0] - self.lat_min) / (self.lat_max - self.lat_min)
        y_norm[:, 1] = (y[:, 1] - self.lon_min) / (self.lon_max - self.lon_min)
        
        return (
            torch.tensor(x_norm, dtype=torch.float32),
            torch.tensor(y_norm, dtype=torch.float32)
        )
    
    def denormalize(self, coords):
        """Convert back to degrees"""
        lat = coords[..., 0] * (self.lat_max - self.lat_min) + self.lat_min
        lon = coords[..., 1] * (self.lon_max - self.lon_min) + self.lon_min
        return lat, lon

# =========================
# SIMPLE MODEL 
# =========================
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=256, num_layers=3, pred_len=4):
        super().__init__()
        
        self.pred_len = pred_len

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, pred_len * 2)
        )
        
    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        return output.view(-1, self.pred_len, 2)

# =========================
# HAVERSINE DISTANCE
# =========================
def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in kilometers"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model = None
    
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            print(f"  ✓ New best model (val_loss: {avg_val_loss:.6f})")
    model.load_state_dict(best_model)
    print(f"\n✓ Loaded best model with val_loss: {best_val_loss:.6f}")
    
    return model

# =========================
# EVALUATION
# =========================
def evaluate_model(model, test_loader, dataset):
    model.eval()
    all_errors = []
    step_errors = [[] for _ in range(PRED_LEN)]
    
    print("\n" + "="*60)
    print("Evaluating on Test Set")
    print("="*60)
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            # Move to CPU
            pred_np = pred.cpu().numpy()
            target_np = y.cpu().numpy()
            
            # Denormalize and compute errors
            for i in range(len(pred_np)):
                lat_pred, lon_pred = dataset.denormalize(pred_np[i])
                lat_true, lon_true = dataset.denormalize(target_np[i])
                
                for j in range(PRED_LEN):
                    error = haversine_distance(
                        lat_pred[j], lon_pred[j],
                        lat_true[j], lon_true[j]
                    )
                    all_errors.append(error)
                    step_errors[j].append(error)
    
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    step_means = [np.mean(step_errors[j]) for j in range(PRED_LEN)]
    
    return mean_error, std_error, step_means, all_errors

# =========================
# VISUALIZATION
# =========================
def plot_results(errors, step_errors):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error distribution
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(np.mean(errors), color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(errors):.1f} km')
    axes[0].set_xlabel('Prediction Error (km)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error by forecast hour
    hours = [6, 12, 18, 24]
    axes[1].bar(hours, step_errors, color='coral', edgecolor='black', alpha=0.7)
    axes[1].plot(hours, step_errors, 'ro-', markersize=8, linewidth=2)
    axes[1].set_xlabel('Forecast Hour')
    axes[1].set_ylabel('Mean Error (km)')
    axes[1].set_title('Error by Forecast Time')
    axes[1].grid(True, alpha=0.3)
    
    for i, (h, e) in enumerate(zip(hours, step_errors)):
        axes[1].text(h, e + 5, f'{e:.0f}km', ha='center', fontsize=10)
    
    plt.suptitle(f'Typhoon Path Prediction - Mean Error: {np.mean(errors):.1f} km',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('typhoon_simple_results.png', dpi=150, bbox_inches='tight')
    plt.show()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("SIMPLE TYPHOON PREDICTION MODEL")
    print("Based on the working model that got 18.82 km")
    print("="*60)
    
    # Create dataset
    dataset = SimpleTyphoonDataset(CSV_PATH, SEQ_LEN, PRED_LEN)
    
    if len(dataset) == 0:
        print("No samples created! Check your data.")
        exit()
    
    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_dataset):,}")
    print(f"  Validation: {len(val_dataset):,}")
    print(f"  Test: {len(test_dataset):,}")
    
    # Create model
    model = SimpleLSTM(
        input_size=2,
        hidden_size=256,
        num_layers=3,
        pred_len=PRED_LEN
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    
    # Train
    model = train_model(model, train_loader, val_loader)
    
    # Evaluate
    mean_error, std_error, step_errors, all_errors = evaluate_model(
        model, test_loader, dataset
    )
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\n 24-hour Prediction Error: {mean_error:.1f} ± {std_error:.1f} km")
    print(f"\n Per-step Performance:")
    for i, hours in enumerate([6, 12, 18, 24]):
        print(f"   {hours}h: {step_errors[i]:.1f} km")
    
    print(f"\n vs Baseline LSTM (81 km):")
    improvement = (81 - mean_error) / 81 * 100
    if mean_error < 81:
        print(f"    SUCCESS! Improvement: {improvement:.1f}%")
        print(f"    Your model beats the LSTM baseline!")
    else:
        print(f"    Current: {mean_error:.1f} km, Target: 81 km")
        print(f"   Need to improve by {mean_error - 81:.1f} km")
    
    # Plot results
    plot_results(all_errors, step_errors)
    
    print("\n Results saved to 'typhoon_simple_results.png'")