# optim.py
def sgd(params, lr=0.01, clip=1.0):
    for p in params:
        p.grad = max(-clip, min(clip, p.grad))
        p.data -= lr * p.grad
        p.grad = 0.0