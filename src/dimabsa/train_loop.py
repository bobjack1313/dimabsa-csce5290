from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from dimabsa.model import SimpleClassifier

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def build_toy_loader(n=256, input_dim=16, num_classes=3, batch_size=32, device=torch.device("cpu")):
    x = torch.randn(n, input_dim)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def train_one_run(epochs: int = 3, input_dim: int = 16, hidden_dim: int = 32, num_classes: int = 3, lr: float = 1e-3, out_dir: Path = Path("experiments/checkpoints")):
    device = get_device()
    model = SimpleClassifier(input_dim, hidden_dim, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    loader = build_toy_loader(input_dim=input_dim, num_classes=num_classes)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        print(f"epoch {epoch} loss={total/len(loader.dataset):.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "simple.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt.resolve()}")
