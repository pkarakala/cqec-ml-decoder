"""
Phase 4: Adaptive GRU Decoder with Online Learning

Implements a GRU decoder that updates its weights during inference using
exponential moving average (EMA) updates. This allows the decoder to adapt
to drifting non-idealities in real-time.

Three adaptation modes:
1. Pure pseudo-label: Uses high-confidence predictions as labels (self-training).
   Fails under heavy drift because confident-but-wrong predictions poison learning.
2. Fully supervised: Uses true labels at every step (oracle, upper bound).
3. Hybrid (periodic supervision): True labels injected every N steps, pseudo-labels
   in between. Models realistic QEC where periodic recalibration is available.

Key finding: Pure self-training fails because of confident wrong predictions,
but periodic recalibration + online adaptation maintains accuracy under drift.
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy


class AdaptiveGRUDecoder(nn.Module):
    """
    GRU decoder with online learning capability.
    
    Architecture is identical to the static GRU, but adds:
    - Exponential moving average (EMA) of gradients
    - Online weight updates during inference
    - Configurable adaptation rate
    """
    
    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 4,
        dropout: float = 0.1,
        # Adaptation parameters
        adapt_lr: float = 0.0001,      # learning rate for online updates
        ema_decay: float = 0.9,         # EMA decay for gradient smoothing
        adapt_every: int = 1,           # update weights every N samples
        confidence_threshold: float = 0.8  # only adapt on high-confidence predictions
    ):
        super().__init__()
        
        self.adapt_lr = adapt_lr
        self.ema_decay = ema_decay
        self.adapt_every = adapt_every
        self.confidence_threshold = confidence_threshold
        
        # Same architecture as static GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # EMA buffers for gradients (initialized on first update)
        self.ema_grads = None
        self.update_count = 0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass (same as static GRU)."""
        output, h_n = self.gru(x)
        last_hidden = h_n[-1]
        logits = self.classifier(last_hidden)
        return logits
    
    def adapt_step(self, x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor | None = None):
        """
        Perform one online adaptation step.
        
        Parameters
        ----------
        x : torch.Tensor
            Input window (batch_size, window_size, 2)
        y_pred : torch.Tensor
            Predicted logits from forward pass
        y_true : torch.Tensor or None
            True labels (if available). If None, uses pseudo-labels from predictions.
        """
        self.update_count += 1
        
        # Only update every N samples
        if self.update_count % self.adapt_every != 0:
            return
        
        # Determine labels for adaptation
        if y_true is None:
            # Use pseudo-labels: only adapt on high-confidence predictions
            probs = torch.softmax(y_pred, dim=1)
            confidence, pseudo_labels = probs.max(dim=1)
            
            # Filter to high-confidence samples
            mask = confidence >= self.confidence_threshold
            if mask.sum() == 0:
                return  # no confident predictions, skip update
            
            x = x[mask]
            pseudo_labels = pseudo_labels[mask]
            labels = pseudo_labels
        else:
            labels = y_true
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        logits = self.forward(x)
        loss = criterion(logits, labels)
        
        # Compute gradients
        loss.backward()
        
        # Initialize EMA buffers on first update
        if self.ema_grads is None:
            self.ema_grads = {}
            for name, param in self.named_parameters():
                if param.grad is not None:
                    self.ema_grads[name] = param.grad.clone().detach()
        
        # Update EMA gradients and apply updates
        with torch.no_grad():
            for name, param in self.named_parameters():
                if param.grad is not None:
                    # EMA update: g_ema = decay * g_ema + (1 - decay) * g_new
                    self.ema_grads[name] = (self.ema_decay * self.ema_grads[name] + 
                                            (1 - self.ema_decay) * param.grad)
                    
                    # Apply smoothed gradient
                    param.data -= self.adapt_lr * self.ema_grads[name]
        
        # Clear gradients for next iteration
        self.zero_grad()
    
    def predict_adaptive(
        self,
        X: np.ndarray,
        y_true: np.ndarray | None = None,
        reset_ema: bool = True,
        supervised_every: int = 0
    ) -> tuple[np.ndarray, dict]:
        """
        Predict with online adaptation.

        Parameters
        ----------
        X : np.ndarray
            Input windows (n_samples, window_size, 2)
        y_true : np.ndarray or None
            True labels. Used for:
            - Full supervised adaptation if supervised_every == 0 and y_true is provided
            - Periodic supervision if supervised_every > 0 (true label injected every N steps)
            - Pure pseudo-label mode if y_true is None
        reset_ema : bool
            Whether to reset EMA buffers before prediction
        supervised_every : int
            If > 0, inject true labels every N steps (hybrid mode).
            Requires y_true to be provided.
            If 0, behavior depends on y_true:
              - y_true provided: fully supervised adaptation (every step)
              - y_true is None: pure pseudo-label adaptation

        Returns
        -------
        predictions : np.ndarray
            Predicted labels (n_samples,)
        history : dict
            Tracking information:
            - 'confidences': prediction confidence at each step
            - 'adapted': whether adaptation occurred at each step
            - 'supervised': whether true label was used at each step
        """
        if supervised_every > 0 and y_true is None:
            raise ValueError("supervised_every > 0 requires y_true to be provided")

        if reset_ema:
            self.ema_grads = None
            self.update_count = 0

        self.train()  # enable gradient computation

        predictions = []
        confidences = []
        adapted = []
        supervised = []

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y_true, dtype=torch.long) if y_true is not None else None

        for i in range(len(X)):
            # Forward pass
            x_i = X_tensor[i:i+1]
            with torch.no_grad():
                logits = self.forward(x_i)
                probs = torch.softmax(logits, dim=1)
                confidence, pred = probs.max(dim=1)

            predictions.append(pred.item())
            confidences.append(confidence.item())

            # Determine if this step uses true labels
            use_true_label = False
            if y_tensor is not None:
                if supervised_every == 0:
                    # Fully supervised mode
                    use_true_label = True
                elif (i + 1) % supervised_every == 0:
                    # Periodic supervision: inject true label every N steps
                    use_true_label = True

            # Adaptation step
            y_i = y_tensor[i:i+1] if use_true_label else None

            # Check if we'll actually adapt
            will_adapt = False
            if use_true_label or confidence.item() >= self.confidence_threshold:
                if (self.update_count + 1) % self.adapt_every == 0:
                    will_adapt = True

            self.adapt_step(x_i, logits, y_i)
            adapted.append(will_adapt)
            supervised.append(use_true_label)

        return np.array(predictions), {
            'confidences': np.array(confidences),
            'adapted': np.array(adapted),
            'supervised': np.array(supervised)
        }


def train_adaptive_gru(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    hidden_size: int = 64,
    # Adaptation parameters
    adapt_lr: float = 0.0001,
    ema_decay: float = 0.9,
    adapt_every: int = 1,
    confidence_threshold: float = 0.8,
    seed: int = 42
) -> dict:
    """
    Train an adaptive GRU decoder.
    
    Training phase is identical to static GRU. The adaptation parameters
    only affect inference behavior.
    
    Returns
    -------
    dict with keys:
        - model: trained AdaptiveGRUDecoder
        - history: training curves
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Build model
    model = AdaptiveGRUDecoder(
        hidden_size=hidden_size,
        adapt_lr=adapt_lr,
        ema_decay=ema_decay,
        adapt_every=adapt_every,
        confidence_threshold=confidence_threshold
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        
        perm = torch.randperm(len(X_train_t))
        X_train_t = X_train_t[perm]
        y_train_t = y_train_t[perm]
        
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i + batch_size]
            batch_y = y_train_t[i:i + batch_size]
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            n_batches += 1
        
        avg_train_loss = train_loss_sum / n_batches
        
        # Validation (no adaptation during training validation)
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss: {avg_train_loss:.4f} | "
                  f"val_loss: {val_loss:.4f} | "
                  f"val_acc: {val_acc:.4f}")
    
    return {"model": model, "history": history}
