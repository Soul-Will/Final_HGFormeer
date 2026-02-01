import csv
from pathlib import Path
from datetime import datetime

class UnifiedTrainingLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "phase", "epoch", "step",

            "ssl_total_loss",
            "ssl_inpaint_loss",
            "ssl_rotation_loss",
            "ssl_contrastive_loss",
            "ssl_marker_loss",

            "train_loss",
            "train_dice_loss",
            "train_focal_loss",
            "train_boundary_loss",
            "loss_alpha",

            "val_loss",
            "val_dice",
            "val_iou",

            "learning_rate",
            "timestamp"
        ]

        if not self.log_path.exists():
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, **kwargs):
        row = {k: kwargs.get(k, None) for k in self.fieldnames}
        row["timestamp"] = datetime.now().isoformat()

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)
