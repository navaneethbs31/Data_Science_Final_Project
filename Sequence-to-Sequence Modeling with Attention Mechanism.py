import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# 1. Generate Synthetic Dataset
def generate_data(num_samples, sequence_length):
    X = np.random.randint(1, 10, size=(num_samples, sequence_length))
    y = np.array([x[::-1] for x in X])
    return X, y


# Parameters
num_samples = 10000
sequence_length = 10
input_dim = output_dim = 10
hidden_dim = 32
embedding_dim = 16

X, y = generate_data(num_samples, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# 2. Implement Seq2Seq Model with Attention Mechanism
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, src, trg):
        src_embedded = self.embedding(src)
        trg_embedded = self.embedding(trg)

        encoder_outputs, (hidden, cell) = self.encoder_lstm(src_embedded)

        attention_weights = torch.bmm(trg_embedded, encoder_outputs.transpose(1, 2))
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=2)

        context = torch.bmm(attention_weights, encoder_outputs)

        decoder_input = torch.cat((trg_embedded, context), dim=2)
        decoder_outputs, _ = self.decoder_lstm(decoder_input, (hidden, cell))

        output = self.fc(decoder_outputs)
        return output


# Instantiate model, loss function, and optimizer
model = Seq2Seq(input_dim, output_dim, hidden_dim, embedding_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 3. Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    loss_history = []
    for epoch in range(epochs):
        total_loss = 0
        for src, trg in train_loader:
            optimizer.zero_grad()
            output = model(src, trg)
            loss = criterion(output.view(-1, output_dim), trg.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
    return loss_history


loss_history = train_model(model, train_loader, criterion, optimizer)


# 4. Evaluate the Model
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in data_loader:
            output = model(src, trg)
            loss = criterion(output.view(-1, output_dim), trg.view(-1))
            total_loss += loss.item()
    return total_loss / len(data_loader)


test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loss = evaluate_model(model, test_loader)
print(f'Test Loss: {test_loss}')


# 5. Plot Loss Curves
def plot_loss_curve(losses, title="Loss Curve"):
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.show()


plot_loss_curve(loss_history)
