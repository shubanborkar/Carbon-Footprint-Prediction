

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Load Dataset
df = pd.read_csv("emissions.csv")

# Enhanced Feature Engineering
categorical_features = ['CompanyID', 'CompanyName', 'Sector', 'Location', 'Year']
numerical_features = ['CH4', 'N2O', 'Electricity', 'FossilFuels', 'Renewables', 
                     'RecycledWaste', 'LandfillWaste', 'CompostedWaste']

# Categorical encoding
for feat in categorical_features:
    df[f'{feat}_encoded'] = df[feat].astype('category').cat.codes.astype(np.float32)  # Ensure float32

# Create interaction features
df['Energy_Mix'] = (df['Electricity'] / (df['FossilFuels'] + df['Renewables'] + 1e-8)).astype(np.float32)
df['Waste_Ratio'] = (df['RecycledWaste'] / (df['LandfillWaste'] + df['CompostedWaste'] + 1e-8)).astype(np.float32)


# Select Features & Target
features = ([f'{feat}_encoded' for feat in categorical_features] + 
           numerical_features + ['Energy_Mix', 'Waste_Ratio'])
target = 'CO2'

# Convert to numpy arrays with explicit dtype
X = df[features].values.astype(np.float32)
y = df[target].values.astype(np.float32).reshape(-1, 1)

# Robust Normalization
def robust_normalize(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-8
    return (data - mean) / std, mean, std

X_normalized, X_mean, X_std = robust_normalize(X)
y_normalized, y_mean, y_std = robust_normalize(y)

# Save normalization parameters
normalization_params = {
    'X_mean': X_mean,
    'X_std': X_std,
    'y_mean': y_mean,
    'y_std': y_std,
    'features': features # Save feature names to ensure correct order
}
joblib.dump(normalization_params, 'normalization_params.pkl')
print("Normalization parameters saved to normalization_params.pkl")

# Train-Validation-Test Split
np.random.seed(42)
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

train_idx = indices[:train_size]
val_idx = indices[train_size:train_size + val_size]
test_idx = indices[train_size + val_size:]

X_train, y_train = X_normalized[train_idx], y_normalized[train_idx]
X_val, y_val = X_normalized[val_idx], y_normalized[val_idx]
X_test, y_test = X_normalized[test_idx], y_normalized[test_idx]

# Convert to PyTorch tensors with explicit dtype
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class EmissionsDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# DataLoaders
batch_size = min(32, len(X_train) // 10)
train_loader = torch.utils.data.DataLoader(EmissionsDataset(X_train_tensor, y_train_tensor), 
                                         batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(EmissionsDataset(X_val_tensor, y_val_tensor), 
                                       batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(EmissionsDataset(X_test_tensor, y_test_tensor), 
                                        batch_size=batch_size, shuffle=False)

class CO2Predictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        return self.network(x)

# Initialize model and training components
model = CO2Predictor(X_train.shape[1])
criterion = nn.HuberLoss(delta=1.0)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                               patience=5, verbose=False)

def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
    return total_loss / len(val_loader)

# Training with early stopping
num_epochs = 200
patience = 15
min_delta = 1e-4
best_val_loss = float('inf')
counter = 0
best_epoch = 0

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    val_loss = validate(model, val_loader, criterion)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), "co2_emission_best.pt")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping triggered. Best epoch was {best_epoch + 1}")
            break

# Evaluate Model
model.load_state_dict(torch.load("co2_emission_best.pt"))
model.eval()

with torch.no_grad():
    test_predictions = model(X_test_tensor).numpy()
    
    # Denormalize predictions and actual values
    test_predictions = test_predictions * y_std + y_mean
    actual_values = y_test * y_std + y_mean
    
    # Calculate metrics
    mse = np.mean((actual_values - test_predictions) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_values - test_predictions))
    r2 = 1 - (np.sum((actual_values - test_predictions) ** 2) / 
              np.sum((actual_values - np.mean(actual_values)) ** 2))
    
    print("\nModel Performance:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"\nSample Predictions:")
    for i in range(min(5, len(test_predictions))):
        print(f"Actual: {actual_values[i][0]:.2f}, Predicted: {test_predictions[i][0]:.2f}, "
              f"Error: {abs(actual_values[i][0] - test_predictions[i][0]):.2f}")