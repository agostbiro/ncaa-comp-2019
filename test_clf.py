import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class Net(nn.Module):
    """Binary classifier for two inputs"""
    def __init__(self, n_classes, n_features, hidden_size):
        super().__init__()
        self.n_classes = n_classes

        self.i2h = nn.Linear(2 * n_features, hidden_size)
        self.h2o = nn.Linear(hidden_size, self.n_classes)

    def forward(self, input1, input2):
        combined = torch.cat((input1, input2), -1)
        h = F.relu(self.i2h(combined))
        pred = F.log_softmax(self.h2o(h), dim=-1)
        return pred


def compute_target(features, f_weights):
    """Compute a non-linear function of the features."""
    return np.linalg.norm(features * f_weights)


def train(model, optimizer, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        optimizer.zero_grad()
        preds = model(features[i, :, 0], features[i, :, 1])
        loss = loss_fn(preds, targets[i])
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return np.mean(losses)


def validate(model, loss_fn, features, targets):
    """Train the model for one epoch."""
    losses = []
    for i in range(features.shape[0]):
        preds = model(features[i, :, 0], features[i, :, 1])
        loss = loss_fn(preds, targets[i])
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

lr = 0.01
l2_reg = 0.01

# Create an MLP for regression, loss and optimizer
model = Net(n_classes, n_features, n_hidden)
loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

model.to(device)

# Create test data
rand = np.random.RandomState(12)
f_shape = (n_batches, batch_size, n_classes, n_features)

features = rand.randint(0, 101, size=f_shape)
f_weights = rand.normal(0, 0.5, size=(n_features,))

targets = np.zeros((n_batches, batch_size))
for i in range(n_batches):
    for j in range(batch_size):
        t1 = compute_target(features[i, j, 0], f_weights)
        t2 = compute_target(features[i, j, 1], f_weights)
        if t1 > t2:
            targets[i, j] = 1

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

