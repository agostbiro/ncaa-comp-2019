import torch
from torch import nn
from torch import optim
import numpy as np


def compute_target(features, f_weights):
    """Compute a non-linear function of the features."""
    return np.linalg.norm(features * f_weights)


def train(model, optimizer, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        optimizer.zero_grad()
        preds = model(features[i])
        loss = loss_fn(preds, targets[i])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def validate(model, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        preds = model(features[i])
        loss = loss_fn(preds, targets[i])
        losses.append(loss.item())
    return np.mean(losses)


# Params
device = 'cuda'

epochs = 100
n_batches = 30
n_val_batch = 5
batch_size = 32
n_features = 8
n_hidden = 8

lr = 0.01
l2_reg = 0.01

# Create an MLP for regression, loss and optimizer
model = nn.Sequential(
    nn.Linear(n_features, n_hidden),
    nn.ReLU(),
    nn.Linear(n_hidden, 1)
)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

model.to(device)

# Create test data
rand = np.random.RandomState(123)
f_shape = (n_batches, batch_size, n_features)

features = rand.randint(0, 101, size=f_shape)
f_weights = rand.normal(0, 0.5, size=(n_features,))

targets = np.zeros((n_batches, batch_size, 1))
for i in range(n_batches):
    for j in range(batch_size):
        targets[i, j, 0] = compute_target(features[i, j],
                                          f_weights)

features = torch.tensor(features, dtype=torch.float, device=device)
targets = torch.tensor(targets, dtype=torch.float, device=device)

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

