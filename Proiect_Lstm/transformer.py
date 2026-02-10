import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_length = 24
batch_size = 32
hidden_dim = 64
n_layers = 2
n_heads = 4
epochs = 50

features = ["Global_active_power", "Global_reactive_power", "Voltage",
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, 0]  # Predict next Power value
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


consumption = pd.read_csv("../household_hourly_dataset.csv")
data_raw = consumption[features].values.astype(float)

split_idx = int(len(data_raw) * 0.7)
train_raw = data_raw[:split_idx]
test_raw = data_raw[split_idx:]

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_raw)
test_scaled = scaler.transform(test_raw)

X_train, y_train = create_sequences(train_scaled, seq_length)
X_test, y_test = create_sequences(test_scaled, seq_length)

X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.FloatTensor(y_test).to(device)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_heads):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_linear(x) + self.positional_encoding
        x = self.transformer_encoder(x)
        last_time_step = x[:, -1, :]
        out = self.output_linear(last_time_step)
        return out


model = TimeSeriesTransformer(input_dim=len(features), hidden_dim=hidden_dim,
                              n_layers=n_layers, n_heads=n_heads).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
loss_vector = np.zeros(epochs)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.5f}")
    loss_vector[epoch] = total_loss / len(train_loader)

model.eval()
with torch.no_grad():
    preds = model(X_test).cpu().numpy()
    actuals = y_test.cpu().numpy()

pred_dummy = np.zeros((len(preds), len(features)))
actual_dummy = np.zeros((len(actuals), len(features)))

pred_dummy[:, 0] = preds.flatten()
actual_dummy[:, 0] = actuals.flatten()

final_preds = scaler.inverse_transform(pred_dummy)[:, 0]
final_actuals = scaler.inverse_transform(actual_dummy)[:, 0]

plt.figure(figsize=(12, 6))
plt.title("Transformer Results")
plt.plot(final_actuals[:200], label="Ground Truth")
plt.plot(final_preds[:200], label="Transformer Prediction", color='red', linestyle=':')
plt.legend()
plt.savefig("Transformer_pred.pdf", format="pdf")

plt.figure(figsize=(12, 6))
plt.title("Transformer Loss")
plt.plot(np.arange(1, epochs + 1, 1), loss_vector, label="Ground Truth")
plt.legend()
plt.savefig("Transformer_loss.pdf", format="pdf")