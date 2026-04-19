
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise SystemExit("PyTorch가 필요합니다. pip install torch 로 설치하세요.") from e


def make_grid(size=32):
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y, indexing="xy")
    return X, Y


def smooth_square_radius(X, Y, sharpness=8.0):
    return (np.abs(X)**sharpness + np.abs(Y)**sharpness) ** (1.0 / sharpness)


def generate_sequence(params, T=8, size=32):
    depth = params["depth"]
    radius = params["radius"]
    friction = params["friction"]
    bhf = params["bhf"]
    anisot = params["anisot"]

    X, Y = make_grid(size=size)
    sq = smooth_square_radius(X, Y, sharpness=radius)
    mask = (sq <= 1.0).astype(np.float32)

    wall = np.clip(1.0 - sq, 0.0, 1.0)
    corner_factor = np.exp(-((np.abs(X) - np.abs(Y))**2) / (0.18 + 0.05 * friction))
    target_shape = depth * (wall ** (1.5 + 0.8 * friction)) * mask

    direction = 1.0 + anisot * (0.6 * X**2 + 0.4 * Y**2)
    friction_bhf = 0.6 + 0.7 * friction + 0.5 * bhf

    seq = []
    for t in range(T):
        alpha = t / (T - 1)
        z = alpha * target_shape
        gy, gx = np.gradient(z)
        gradient = np.sqrt(np.maximum(gy**2 + gx**2, 0))
        eps = friction_bhf * alpha * (0.8 * wall + 0.7 * gradient) * direction * mask
        th = np.clip(1.0 - 0.22 * eps - 0.05 * alpha * corner_factor * mask, 0.65, 1.05)
        seq.append(np.stack([z, th, eps], axis=0).astype(np.float32))
    seq = np.stack(seq, axis=0)
    cond = np.array([depth, radius, friction, bhf, anisot], dtype=np.float32)
    return cond, seq


class ToyPressDataset(Dataset):
    def __init__(self, n_cases=120, T=8, size=32, seed=0):
        rng = np.random.default_rng(seed)
        self.samples = []
        for _ in range(n_cases):
            params = {
                "depth": rng.uniform(0.35, 0.85),
                "radius": rng.uniform(6.0, 14.0),
                "friction": rng.uniform(0.05, 0.35),
                "bhf": rng.uniform(0.10, 0.90),
                "anisot": rng.uniform(-0.20, 0.20),
            }
            cond, seq = generate_sequence(params, T=T, size=size)
            for t in range(T - 1):
                self.samples.append((cond, seq[t], seq[t + 1]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond, cur_state, next_state = self.samples[idx]
        return (
            torch.tensor(cond, dtype=torch.float32),
            torch.tensor(cur_state, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
        )


class TinyWorldModel(nn.Module):
    def __init__(self, h=32, w=32, cond_dim=5, hidden=16):
        super().__init__()
        self.h = h
        self.w = w
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, 24),
            nn.ReLU(),
            nn.Linear(24, hidden),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Conv2d(3 + hidden, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 3, kernel_size=1),
        )

    def forward(self, cur_state, cond):
        b = cur_state.shape[0]
        c = self.cond_mlp(cond).view(b, -1, 1, 1).expand(b, -1, self.h, self.w)
        x = torch.cat([cur_state, c], dim=1)
        return self.net(x)


def train_model(device="cpu", epochs=4, batch_size=32, lr=1e-3, size=32, T=8):
    train_ds = ToyPressDataset(n_cases=120, T=T, size=size, seed=0)
    val_ds = ToyPressDataset(n_cases=20, T=T, size=size, seed=123)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TinyWorldModel(h=size, w=size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    history = {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for cond, cur_state, next_state in train_loader:
            cond = cond.to(device)
            cur_state = cur_state.to(device)
            next_state = next_state.to(device)
            pred = model(cur_state, cond)
            loss = loss_fn(pred, next_state)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * cond.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cond, cur_state, next_state in val_loader:
                cond = cond.to(device)
                cur_state = cur_state.to(device)
                next_state = next_state.to(device)
                pred = model(cur_state, cond)
                val_loss += loss_fn(pred, next_state).item() * cond.size(0)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} | train {train_loss:.6f} | val {val_loss:.6f}")

    return model, history


def rollout(model, cond, init_state, steps, device="cpu"):
    model.eval()
    states = [init_state.copy()]
    cur = torch.tensor(init_state[None], dtype=torch.float32).to(device)
    c = torch.tensor(cond[None], dtype=torch.float32).to(device)
    with torch.no_grad():
        for _ in range(steps - 1):
            nxt = model(cur, c)
            cur = nxt
            states.append(nxt[0].cpu().numpy())
    return np.stack(states, axis=0)


def plot_results(cond, true_seq, pred_seq, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    step_ids = [0, len(true_seq)//2, len(true_seq)-1]
    names = ["초기", "중간", "최종"]
    field_names = ["형상 z", "두께 th", "등가변형률 eps"]

    fig, axes = plt.subplots(6, len(step_ids), figsize=(10, 10))
    for j, s in enumerate(step_ids):
        for i in range(3):
            ax = axes[i, j]
            im = ax.imshow(true_seq[s, i], origin="lower")
            ax.set_title(f"정답 {names[j]} - {field_names[i]}")
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            ax2 = axes[i+3, j]
            im2 = ax2.imshow(pred_seq[s, i], origin="lower")
            ax2.set_title(f"예측 {names[j]} - {field_names[i]}")
            ax2.set_xticks([]); ax2.set_yticks([])
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Tiny World Model rollout 예시\n"
        f"depth={cond[0]:.2f}, radius={cond[1]:.2f}, friction={cond[2]:.2f}, "
        f"BHF={cond[3]:.2f}, anisot={cond[4]:.2f}",
        fontsize=11
    )
    fig.tight_layout()
    fig.savefig(outdir / "toy_world_model_rollout.png", dpi=170)
    plt.close(fig)


def plot_history(history, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3.5))
    plt.plot(history["train"], label="train")
    plt.plot(history["val"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.title("Tiny World Model training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "toy_world_model_history.png", dpi=170)
    plt.close()


def main():
    outdir = Path("toy_world_model_outputs")
    outdir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    T = 8
    size = 32
    model, history = train_model(device=device, epochs=4, batch_size=32, lr=1e-3, size=size, T=T)
    plot_history(history, outdir)

    test_params = {
        "depth": 0.78,
        "radius": 10.5,
        "friction": 0.22,
        "bhf": 0.65,
        "anisot": 0.12,
    }
    cond, true_seq = generate_sequence(test_params, T=T, size=size)
    pred_seq = rollout(model, cond, true_seq[0], steps=T, device=device)
    plot_results(cond, true_seq, pred_seq, outdir)
    torch.save(model.state_dict(), outdir / "tiny_world_model.pt")

    print("\n완료 파일:")
    print(outdir / "toy_world_model_history.png")
    print(outdir / "toy_world_model_rollout.png")
    print(outdir / "tiny_world_model.pt")


if __name__ == "__main__":
    main()
