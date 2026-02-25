# main.py
from baifs.models import MLP, MicroResNet
from baifs.solver import train, validate
from baifs.data import mlp_dataset, cnn_dataset, mlp_test_dataset, cnn_test_dataset
from baifs.losses import mse_loss
from baifs.infer import test

def run_mlp():
    X, y = mlp_dataset()
    model = MLP(input_size=3, hidden_size=4, output_size=2)
    print("=== Training MLP ===")
    train(model=model, X_train=X, y_train=y, epochs=20, lr=0.01, verbose=True)

    X_test, y_test = mlp_test_dataset()
    print("\n=== Validating MLP ===")
    validate(model=model, X=X_test, Y=y_test, loss_fn=mse_loss)

    print("\n=== MLP Inference ===")
    preds = test(model, X_test)
    for i, p in enumerate(preds):
        print(f"  Sample {i}: {p}")

def run_cnn():
    H, W, in_ch = 4, 4, 1
    X, y = cnn_dataset(H, W)
    model = MicroResNet(in_ch=in_ch, spatial_h=H, spatial_w=W, num_classes=2)
    print("\n=== Training MicroResNet ===")
    train(model=model, X_train=X, y_train=y, epochs=20, lr=0.001, verbose=True)

    X_test, y_test = cnn_test_dataset(H, W)
    print("\n=== Validating MicroResNet ===")
    validate(model=model, X=X_test, Y=y_test, loss_fn=mse_loss)

    print("\n=== MicroResNet Inference ===")
    preds = test(model, X_test)
    for i, p in enumerate(preds):
        print(f"  Sample {i}: {p}")

if __name__ == "__main__":
    run_mlp()
    run_cnn()