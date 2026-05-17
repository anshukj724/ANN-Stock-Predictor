
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)


# DATA

df = yf.download('AAPL', period='5y', auto_adjust=True)
df.columns = df.columns.get_level_values(0)
print(f"Downloaded: {df.shape}")

# Returns
df['returns'] = df['Close'].pct_change()
df['log'] = np.log(df['Close'] / df['Close'].shift(1))

# Moving averages
df['ma20'] = df['Close'].rolling(20).mean()
df['ma50'] = df['Close'].rolling(50).mean()
df['ma200'] = df['Close'].rolling(200).mean()
df['ma20_50_cross'] = df['ma20'] - df['ma50']
df['ma50_200_cross'] = df['ma50'] - df['ma200']

# Volatility
df['volatility'] = df['Close'].rolling(20).std()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
df['RSI'] = 100 - (100 / (1 + gain/loss))

# Bollinger Bands
df['bb_upper'] = df['ma20'] + 2 * df['volatility']
df['bb_lower'] = df['ma20'] - 2 * df['volatility']
df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

# MACD
exp12 = df['Close'].ewm(span=12, adjust=False).mean()
exp26 = df['Close'].ewm(span=26, adjust=False).mean()
df['macd'] = exp12 - exp26
df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

# Volume
df['volume_z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / df['Volume'].rolling(20).std()

# Target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)


# FEATURES (derived signals only - no raw prices)

base_features = [
    'returns',
    'log',
    'ma20_50_cross',
    'ma50_200_cross',
    'volatility',
    'RSI',
    'bb_position',
    'macd',
    'macd_signal',
    'volume_z'
]

# Add lagged features
num_lags = 3
features = []
for feature in base_features:
    features.append(feature)
    for lag in range(1, num_lags + 1):
        df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
        features.append(f'{feature}_lag{lag}')

df.dropna(inplace=True)
print(f"After cleaning and lagging: {df.shape}")
print(f"UP days: {df['target'].sum()} DOWN days: {(df['target']==0).sum()}")

X = df[features]
y = df['target']


# Split into training and test sets first
train_test_split_idx = int(len(df) * 0.8)
X_train_val = X.iloc[:train_test_split_idx]
X_test      = X.iloc[train_test_split_idx:]
y_train_val = y.iloc[:train_test_split_idx]
y_test      = y.iloc[train_test_split_idx:]

# Further split training data into actual training and validation sets
train_val_split_idx = int(len(X_train_val) * 0.85)
X_train = X_train_val.iloc[:train_val_split_idx]
X_val   = X_train_val.iloc[train_val_split_idx:]
y_train = y_train_val.iloc[:train_val_split_idx]
y_val   = y_train_val.iloc[train_val_split_idx:]

print(f"Train: {len(X_train)} Validation: {len(X_val)} Test: {len(X_test)}")

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


# TENSORS
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train.values.copy()).view(-1, 1)
X_val_tensor   = torch.FloatTensor(X_val_scaled.copy())
y_val_tensor   = torch.FloatTensor(y_val.values).view(-1, 1)
X_test_tensor  = torch.FloatTensor(X_test_scaled.copy())
y_test_tensor  = torch.FloatTensor(y_test.values).view(-1, 1)


# MODEL

class StockANN(nn.Module):
    def __init__(self, input_size):
        super(StockANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = StockANN(input_size=len(features)) # Adjusted input_size
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# TRAINING
losses = []
val_losses = []
best_val_loss = float('inf')
epochs_no_improve = 0
patience = 20 # Number of epochs to wait for improvement before stopping

for epoch in range(300):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    # Validation step
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_losses.append(val_loss.item())

    if epoch % 30 == 0:
        print(f"Epoch {epoch} Train Loss: {loss.item():.4f} Val Loss: {val_loss.item():.4f}")

    # Early stopping logic
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        epochs_no_improve = 0
        # Save the best model state
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping at epoch {epoch} as validation loss did not improve for {patience} epochs.")
            break

# Load the best model before evaluation
model.load_state_dict(torch.load('best_model.pth'))


# EVALUATION

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    predicted_classes = (predictions > 0.5).float()

print(f"\nPredicted UP:   {int(predicted_classes.sum())}")
print(f"Predicted DOWN: {int((predicted_classes==0).sum())}")

accuracy = accuracy_score(
    y_test_tensor.numpy(),
    predicted_classes.numpy()
)

# Adjusted actual_returns to match the test set after new split
actual_returns = df['returns'].iloc[train_test_split_idx:].values
pred_flat = predicted_classes.numpy().flatten()
strategy_returns = actual_returns * np.where(pred_flat == 1, 1, -1)

sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252)
bh_sharpe = (actual_returns.mean() / actual_returns.std()) * np.sqrt(252)

cumulative = (1 + strategy_returns).cumprod()
rolling_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print(f"\n{'='*40}")
print(f"ACCURACY:          {accuracy:.2%}")
print(f"STRATEGY SHARPE:   {sharpe:.2f}")
print(f"BUY & HOLD SHARPE: {bh_sharpe:.2f}")
print(f"MAX DRAWDOWN:      {max_drawdown:.2%}")
print(f"{'='*40}")


# PLOTS

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(losses, label='Training Loss')
axes[0].plot(val_losses, label='Validation Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

cumulative_strategy = (1 + strategy_returns).cumprod()
cumulative_bh = (1 + actual_returns).cumprod()
axes[1].plot(cumulative_strategy, label='ANN Strategy', color='blue')
axes[1].plot(cumulative_bh, label='Buy & Hold', color='green')
axes[1].legend()
axes[1].set_title('Strategy vs Buy & Hold')
axes[1].set_ylabel('Portfolio Value')

axes[2].fill_between(range(len(drawdown)), drawdown, color='red', alpha=0.4)
axes[2].set_title('Strategy Drawdown')
axes[2].set_ylabel('Drawdown')

plt.tight_layout()
plt.savefig('aapl_results.png', dpi=150, bbox_inches='tight')
plt.show()