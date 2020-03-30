import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Net(nn.Module):
    """Binary classifier for items in a sequence."""
    def __init__(self, n_classes, n_features, hidden_size):
        super().__init__()
        self.n_classes = n_classes

        self.rnn = nn.RNN(n_features, hidden_size, nonlinearity='relu')

        self.o2p = nn.Linear(hidden_size, self.n_classes)

    def forward(self, input):
        output, hidden = self.rnn(input)
        pred = F.log_softmax(self.o2p(output), dim=-1)
        return pred


def compute_target(features, f_weights):
    """Compute a non-linear function of the features."""
    z = features * f_weights
    return np.linalg.norm(np.where(z > 0.5 * np.sum(z) / np.size(z), z, 0))


def train(model, optimizer, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        optimizer.zero_grad()
        preds = model(features[i])
        loss = loss_fn(preds.permute(1, 2, 0), targets[i])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def validate(model, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        preds = model(features[i])
        print(preds.size(), targets[i].size())
        loss = loss_fn(preds.permute(1, 2, 0), targets[i])
        losses.append(loss.item())
    return np.mean(losses)


# Params
device = 'cuda'

epochs = 200
n_batches = 30
n_val_batch = 5
batch_size = 32
n_features = 8
n_hidden = 8
n_classes = 2
seq_len = 6

lr = 0.01
l2_reg = 0.01

# Create an MLP for regression, loss and optimizer
model = Net(n_classes, n_features, n_hidden)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

model.to(device)

# Create test data
rand = np.random.RandomState(12)
f_shape = (n_batches, seq_len, batch_size, n_features)

features = rand.randint(0, 101, size=f_shape)
f_weights = rand.normal(0, 0.5, size=(n_features,))

targets = np.zeros((n_batches, batch_size, seq_len))
for i in range(n_batches):
    for j in range(batch_size):
        sum1, sum2 = 0, 0
        for k in range(seq_len):
            sum1 += compute_target(features[i, k, j], f_weights)
            sum2 += compute_target(features[i, k, j], f_weights)
            if sum1 > sum2:
                targets[i, j, k] = 1

features = torch.tensor(features, dtype=torch.float, device=device)
targets = torch.tensor(targets, dtype=torch.long, device=device)

# Train/validation split
train_features = features[:-n_val_batch]
train_targets = targets[:-n_val_batch]
val_features = features[-n_val_batch:]
val_targets = targets[-n_val_batch:]

# Capture pre-train validation loss
pre_val_loss = validate(model, loss_fn, val_features, val_targets)

# Train network
for i in range(epochs):
    loss = train(model, optimizer, loss_fn, train_features, train_targets)
    if i % 10 == 0 or i == epochs - 1:
        print(i, loss)

# Run validation
val_loss = validate(model, loss_fn, val_features, val_targets)
val_dec = round((1 - val_loss / pre_val_loss) * 100, 2)
print('Validation loss decreased after training by {}%'.format(val_dec))

