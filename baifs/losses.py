# losses.py

def mse_loss(y_pred, y_true):
    diff = y_pred - y_true
    sq = diff * diff
    total = sq.sum()
    return total / float(len(y_pred))