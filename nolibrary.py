import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import math

# Load dataset
df = pd.read_csv('emissions.csv')

# Check for duplicate data and clean up
df = df.drop_duplicates()

# Print dataset info
print("Dataset shape:", df.shape)
print("Unique companies:", len(df['CompanyName'].unique()))
print("Unique years:", len(df['Year'].unique()))

# Create one-hot encoding function
def one_hot_encode(df, column):
    unique_values = df[column].unique()
    result = {}
    for i, value in enumerate(unique_values):
        # Skip first value for reference (to avoid perfect collinearity)
        if i == 0:
            continue
        result[f"{column}_{value}"] = (df[column] == value).astype(float)
    return result

# Create our standardizer class
class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Prevent division by zero
        self.std = np.where(self.std == 0, 1, self.std)
        return self
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return data * self.std + self.mean

# Generate one-hot encoded features
categorical_cols = ['CompanyName', 'Sector', 'Location']
numerical_cols = ['Year', 'Electricity', 'FossilFuels', 'Renewables']

# Process each categorical column
encoded_features = {}
for col in categorical_cols:
    encoded_features.update(one_hot_encode(df, col))

# Create processed dataframe
df_processed = pd.DataFrame()

# Add encoded categorical features
for col_name, col_data in encoded_features.items():
    df_processed[col_name] = col_data

# Add numerical features
for col in numerical_cols:
    df_processed[col] = df[col]

# Target variables
y = df[['CO2', 'CH4', 'N2O']].values

# Feature matrix
X = df_processed.values

# Implement train-test split
def train_test_split(X, y, test_size=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    indices = np.random.permutation(len(X))
    test_size_count = int(len(X) * test_size)
    test_indices = indices[:test_size_count]
    train_indices = indices[test_size_count:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)

# Standardize features
scaler_X = Standardizer()
X_train = scaler_X.fit(X_train).transform(X_train)
X_test = scaler_X.transform(X_test)

# Standardize targets
scaler_y = Standardizer()
y_train = scaler_y.fit(y_train).transform(y_train)
y_test = scaler_y.transform(y_test)

# Neural network architecture
input_size = X_train.shape[1]
hidden_size1 = 24
hidden_size2 = 12
output_size = y_train.shape[1]

# Xavier/Glorot initialization
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / (input_size + hidden_size1))
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / (hidden_size1 + hidden_size2))
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2 / (hidden_size2 + output_size))
b3 = np.zeros((1, output_size))

# Activation functions
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Implement mean squared error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Implement R² score
def r2_score(y_true, y_pred):
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2, axis=0)
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    r2 = 1 - (ss_res / ss_tot)
    return r2

# Training parameters
learning_rate = 0.005
epochs = 1000
batch_size = 4
early_stopping_patience = 50
best_loss = float('inf')
patience_counter = 0

# Training loop with early stopping
losses = []
val_losses = []

for epoch in range(epochs):
    # Shuffle data
    indices = np.random.permutation(X_train.shape[0])
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    epoch_losses = []
    for i in range(0, X_train.shape[0], batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        Z1 = np.dot(X_batch, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = relu(Z2)
        Z3 = np.dot(A2, W3) + b3
        y_pred = Z3
        
        # Compute loss (MSE)
        loss = np.mean((y_batch - y_pred) ** 2)
        epoch_losses.append(loss)
        
        # Backpropagation
        dZ3 = 2 * (y_pred - y_batch) / batch_size
        dW3 = np.dot(A2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, W3.T)
        dZ2 = dA2 * relu_derivative(Z2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * relu_derivative(Z1)
        dW1 = np.dot(X_batch.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        
        # Update weights with learning rate decay
        lr = learning_rate / (1 + 0.01 * epoch)
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        W3 -= lr * dW3
        b3 -= lr * db3
    
    # Calculate training loss
    train_loss = np.mean(epoch_losses)
    losses.append(train_loss)
    
    # Calculate validation loss
    Z1_val = np.dot(X_test, W1) + b1
    A1_val = relu(Z1_val)
    Z2_val = np.dot(A1_val, W2) + b2
    A2_val = relu(Z2_val)
    y_val_pred = np.dot(A2_val, W3) + b3
    val_loss = np.mean((y_test - y_val_pred) ** 2)
    val_losses.append(val_loss)
    
    # Early stopping check
    if val_loss < best_loss:
        best_loss = val_loss
        best_weights = (W1.copy(), b1.copy(), W2.copy(), b2.copy(), W3.copy(), b3.copy())
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch}")
        W1, b1, W2, b2, W3, b3 = best_weights
        break
    
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Save the trained model
with open('emissions_model_custom.pkl', 'wb') as f:
    pickle.dump((W1, b1, W2, b2, W3, b3, scaler_X, scaler_y), f)

# Make predictions on test data
Z1 = np.dot(X_test, W1) + b1
A1 = relu(Z1)
Z2 = np.dot(A1, W2) + b2
A2 = relu(Z2)
y_pred = np.dot(A2, W3) + b3

# Denormalize predictions
y_pred_original = scaler_y.inverse_transform(y_pred)
y_test_original = scaler_y.inverse_transform(y_test)

# Calculate metrics
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2, axis=0)

# Calculate metrics for each emission type
mse_co2 = mean_squared_error(y_test_original[:, 0], y_pred_original[:, 0])
r2_co2 = r2_score(y_test_original[:, 0].reshape(-1, 1), y_pred_original[:, 0].reshape(-1, 1))[0]
mse_ch4 = mean_squared_error(y_test_original[:, 1], y_pred_original[:, 1])
r2_ch4 = r2_score(y_test_original[:, 1].reshape(-1, 1), y_pred_original[:, 1].reshape(-1, 1))[0]
mse_n2o = mean_squared_error(y_test_original[:, 2], y_pred_original[:, 2])
r2_n2o = r2_score(y_test_original[:, 2].reshape(-1, 1), y_pred_original[:, 2].reshape(-1, 1))[0]

print(f"CO2 - MSE: {mse_co2:.4f}, R²: {r2_co2:.4f}")
print(f"CH4 - MSE: {mse_ch4:.4f}, R²: {r2_ch4:.4f}")
print(f"N2O - MSE: {mse_n2o:.4f}, R²: {r2_n2o:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(len(losses)), losses, label='Training Loss')
plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('emissions_training_validation_loss_custom.png')
plt.close()

# Function to predict emissions for new data
def predict_emissions(data, model_file='emissions_model_custom.pkl'):
    with open(model_file, 'rb') as f:
        W1, b1, W2, b2, W3, b3, scaler_X, scaler_y = pickle.load(f)
    
    # Preprocess input data
    data_scaled = scaler_X.transform(data)
    
    # Forward pass
    Z1 = np.dot(data_scaled, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    y_pred = np.dot(A2, W3) + b3
    
    # Denormalize predictions
    return scaler_y.inverse_transform(y_pred)

# Generate predictions for the original data to visualize
all_X = df_processed.values
all_X_scaled = scaler_X.transform(all_X)

# Forward pass for all data
Z1_all = np.dot(all_X_scaled, W1) + b1
A1_all = relu(Z1_all)
Z2_all = np.dot(A1_all, W2) + b2
A2_all = relu(Z2_all)
all_preds_scaled = np.dot(A2_all, W3) + b3
all_preds = scaler_y.inverse_transform(all_preds_scaled)

# Add predictions to original dataframe
df['Predicted_CO2'] = all_preds[:, 0]
df['Predicted_CH4'] = all_preds[:, 1]
df['Predicted_N2O'] = all_preds[:, 2]

# Plot actual vs predicted for each emission type (using matplotlib instead of seaborn)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Function to create regression plot without seaborn
def plot_regression(ax, x, y, title, color):
    ax.scatter(x, y, alpha=0.5, color=color)
    
    # Calculate and plot regression line
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, color='red', linewidth=2)
    
    ax.set_xlabel(f"Actual {title}")
    ax.set_ylabel(f"Predicted {title}")
    ax.set_title(f"{title} Emissions: Actual vs. Predicted (R² = {r2_co2:.4f})")
    ax.grid(True, linestyle='--', alpha=0.7)

# CO2 Plot
plot_regression(axes[0], df['CO2'], df['Predicted_CO2'], "CO2", 'blue')

# CH4 Plot
plot_regression(axes[1], df['CH4'], df['Predicted_CH4'], "CH4", 'green')

# N2O Plot
plot_regression(axes[2], df['N2O'], df['Predicted_N2O'], "N2O", 'purple')

plt.tight_layout()
plt.savefig('emissions_predictions_custom.png')
plt.close()

# Feature importance analysis (using weights as a proxy)
feature_names = list(df_processed.columns)
importance_w1 = np.sum(np.abs(W1), axis=1)
feature_importance = [(feature_names[i], importance_w1[i]) for i in range(len(feature_names))]
feature_importance.sort(key=lambda x: x[1], reverse=True)
top_features = feature_importance[:10]

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.barh([f[0] for f in top_features], [f[1] for f in top_features])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features for Emissions Prediction')
plt.tight_layout()
plt.savefig('emissions_feature_importance_custom.png')
plt.close()

print("Model training and evaluation complete. Check the generated plots for visualizations.")