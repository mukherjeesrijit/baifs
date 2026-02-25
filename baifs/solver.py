# solver.py
# solver.py

from .losses import mse_loss
from .optim import sgd

def train(model, X_train, y_train, epochs=20, lr=0.1, verbose=True):
    for epoch in range(epochs):
        total_loss = 0.0
        params = model.parameters()
        for x, y_true in zip(X_train, y_train):
            for p in params:
                p.grad = 0.0
            y_pred = model.forward(x)
            loss = mse_loss(y_pred, y_true)
            total_loss += loss.data
            loss.backward()
            sgd(params, lr=lr)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

def validate(model, X, Y, loss_fn):
    total_loss = 0
    for x, y in zip(X, Y):
        y_pred = model.forward(x)
        loss = loss_fn(y_pred, y)
        total_loss += loss.data
    print(f"Validation Loss: {total_loss:.4f}")