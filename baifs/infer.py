# infer.py
def test(model, X):
    preds = []
    for x in X:
        y_pred = model.forward(x)
        preds.append(y_pred)
    return preds
