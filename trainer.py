import os
import torch

class AdapterTrainer:
    def __init__(self, adapter, loss_fn, config, device="cuda"):
        self.adapter = adapter.to(device)
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=config["train"]["learning_rate"],
            weight_decay=config["train"]["weight_decay"]
        )

    def train_epoch(self, dataloader):
        self.adapter.train()
        total_loss = 0.0

        for base_emb, labels in dataloader:
            base_emb = base_emb.to(self.device)
            labels   = labels.to(self.device)

            adapted = self.adapter(base_emb)
            loss = self.loss_fn(
                adapted=adapted,
                base_emb=base_emb,
                labels=labels,
                **self.config.get("loss", {})
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def fit(self, dataloader):
        for epoch in range(self.config["train"]["epochs"]):
            avg_loss = self.train_epoch(dataloader)
            print(f"Epoch {epoch+1:03d} | loss: {avg_loss:.4f}")

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.adapter.state_dict(), path)