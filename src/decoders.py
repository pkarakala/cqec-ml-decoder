import numpy as np
import torch
import torch.nn as nn


# ─── Decoder A: Threshold Baseline ────────────────────────────
# The simplest possible decoder. No learning involved.
# It just looks at the average of the last W measurements
# and picks the error class based on which quadrant (S1, S2)
# the averages land in. This is your "dumb but honest"
# baseline that the RNN has to beat.
# ───────────────────────────────────────────────────────────────

class ThresholdDecoder:
    """
    Decodes error class by averaging r1 and r2 over the window
    and checking the sign of each average.

    Sign mapping (matches ERROR_SIGNATURES):
        r1_avg > 0 and r2_avg > 0  →  error 0 (no error)
        r1_avg < 0 and r2_avg > 0  →  error 1 (flip on qubit 1)
        r1_avg < 0 and r2_avg < 0  →  error 2 (flip on qubit 2)
        r1_avg > 0 and r2_avg < 0  →  error 3 (flip on qubit 3)
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: array of shape (n_samples, window_size, 2)
           where [:,:,0] = r1 windows, [:,:,1] = r2 windows

        Returns: array of shape (n_samples,) with predicted error labels
        """
        # Average over the time axis (axis=1) for each sample
        r1_avg = X[:, :, 0].mean(axis=1)  # shape: (n_samples,)
        r2_avg = X[:, :, 1].mean(axis=1)  # shape: (n_samples,)

        # Start with all predictions as 0 (no error)
        preds = np.zeros(len(X), dtype=int)

        # Assign error class based on sign combination
        preds[(r1_avg < 0) & (r2_avg >= 0)] = 1  # qubit 1 flipped
        preds[(r1_avg < 0) & (r2_avg < 0)]  = 2  # qubit 2 flipped
        preds[(r1_avg >= 0) & (r2_avg < 0)] = 3  # qubit 3 flipped

        return preds


# ─── Decoder B: GRU (the ML decoder) ──────────────────────────
# A Gated Recurrent Unit network. GRUs are designed for
# sequential data — they process the measurement window
# one timestep at a time, building up an internal "memory"
# of what the signal has been doing. At the end of the window,
# that memory gets fed into a classifier head that outputs
# the most likely error class.
#
# Why GRU and not LSTM? GRU has fewer parameters, trains
# faster, and works just as well on short sequences like ours.
# ───────────────────────────────────────────────────────────────

class GRUDecoder(nn.Module):
    def __init__(
        self,
        input_size: int = 2,       # r1 and r2
        hidden_size: int = 64,     # internal memory size of the GRU
        num_layers: int = 1,       # how many GRU layers stacked
        num_classes: int = 4,      # output classes: 0, 1, 2, 3
        dropout: float = 0.1       # regularization to prevent overfitting
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # input shape is (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0  # dropout only between layers
        )

        # Classifier head: takes the final hidden state and maps to 4 classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape (batch_size, window_size, 2)

        Returns: tensor of shape (batch_size, num_classes)
                 raw logits (not probabilities yet)
        """
        # Run the GRU over the sequence
        # output shape: (batch, seq_len, hidden_size) — all timestep outputs
        # h_n shape:    (num_layers, batch, hidden_size) — final hidden state
        output, h_n = self.gru(x)

        # We only care about the LAST hidden state — that's the summary
        # of the entire sequence. Take the top layer's final state.
        last_hidden = h_n[-1]  # shape: (batch, hidden_size)

        # Feed into classifier to get logits
        logits = self.classifier(last_hidden)  # shape: (batch, num_classes)

        return logits


# ─── Training Loop ────────────────────────────────────────────
# Handles the actual training of the GRU. Wraps the standard
# PyTorch pattern: forward pass → compute loss → backprop → update.
# ───────────────────────────────────────────────────────────────

def train_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    hidden_size: int = 64,
    seed: int = 42
) -> dict:
    """
    Trains the GRU decoder and returns the trained model + history.

    Returns:
        - model:          the trained GRUDecoder
        - history:        dict with 'train_loss', 'val_loss', 'val_acc' per epoch
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Build model, loss, optimizer ──────────────────────────
    model     = GRUDecoder(hidden_size=hidden_size)
    criterion = nn.CrossEntropyLoss()          # standard for multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Convert numpy arrays to PyTorch tensors ──────────────
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,   dtype=torch.long)

    # ── Training history tracking ─────────────────────────────
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        # ── TRAINING ────────────────────────────────────────────
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        # Shuffle training data each epoch
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]

        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]

            optimizer.zero_grad()           # clear old gradients
            logits = model(batch_X)         # forward pass
            loss   = criterion(logits, batch_y)  # compute loss
            loss.backward()                 # backprop
            optimizer.step()                # update weights

            train_loss_sum += loss.item()
            n_batches += 1

        avg_train_loss = train_loss_sum / n_batches

        # ── VALIDATION ──────────────────────────────────────────
        model.eval()
        with torch.no_grad():                       # no gradient tracking needed
            val_logits = model(X_val_t)
            val_loss   = criterion(val_logits, y_val_t).item()
            val_preds  = val_logits.argmax(dim=1)   # pick the highest logit
            val_acc    = (val_preds == y_val_t).float().mean().item()

        # ── Log this epoch ──────────────────────────────────────
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss: {avg_train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | "
                  f"val_acc: {val_acc:.4f}")

    return {"model": model, "history": history}