import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

CONFIG = {
    "K_SHARDS": 20,
    "S_SLICES": 3,
    "EPOCHS_PER_SLICE": 10,
    "BATCH_SIZE": 64,
    "LR": 0.01,
    "UNLEARN_PCTS": [0.01, 0.05],
    "SEED": 42,
    "DELETION_MODE": "recent",
}

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Running SISA on {DEVICE}")


def get_memory_mb():
    """Returns allocated memory (MPS) or RSS (CPU). Note: Not strict peak."""
    if DEVICE.type == "mps":
        return torch.mps.current_allocated_memory() / (1024**2)
    else:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 x 32 x 32 input
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 64 x 16 x 16
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 128 x 8 x 8
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 128 x 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # -> 128*4*4 = 2048
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class SISAEngine:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.shards = {}
        self.checkpoints = {}
        self.shard_models = {}
        self.indices_map = {}

        # Backup storage for independent experiments
        self._backup_shards = None
        self._backup_checkpoints = None
        self._backup_models = None
        self._backup_map = None

        self._partition_data()

    def _partition_data(self):
        all_indices = np.arange(len(self.train_set))
        np.random.shuffle(all_indices)

        shard_splits = np.array_split(all_indices, CONFIG["K_SHARDS"])
        for k, k_indices in enumerate(shard_splits):
            self.shards[k] = {}
            slice_splits = np.array_split(k_indices, CONFIG["S_SLICES"])
            for s, s_indices in enumerate(slice_splits):
                self.shards[k][s] = s_indices
                for idx in s_indices:
                    self.indices_map[idx] = (k, s)

    def backup_state(self):
        self._backup_shards = copy.deepcopy(self.shards)
        self._backup_checkpoints = copy.deepcopy(self.checkpoints)
        self._backup_models = copy.deepcopy(self.shard_models)
        self._backup_map = copy.deepcopy(self.indices_map)

    def restore_state(self):
        self.shards = copy.deepcopy(self._backup_shards)
        self.checkpoints = copy.deepcopy(self._backup_checkpoints)
        self.shard_models = copy.deepcopy(self._backup_models)
        self.indices_map = copy.deepcopy(self._backup_map)

    def _get_loader(self, indices):
        if len(indices) == 0:
            raise ValueError("Attempted to create DataLoader with 0 indices.")
        return DataLoader(
            Subset(self.train_set, indices),
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=True,
        )

    def train_shard_incremental(self, k, start_slice=0):
        model = SimpleCNN().to(DEVICE)
        optimizer = optim.SGD(
            model.parameters(), lr=CONFIG["LR"], momentum=0.9, weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()

        # 1. Load previous state (Isolation & Continuity)
        if start_slice > 0:
            prev_state = self.checkpoints[k][start_slice - 1]
            model.load_state_dict(prev_state)

        # 2. Train incremental slices
        for s in range(start_slice, CONFIG["S_SLICES"]):
            indices = self.shards[k][s]

            # --- NEW: handle empty slice ---
            if len(indices) == 0:
                # No data to train on; just checkpoint the current weights
                if k not in self.checkpoints:
                    self.checkpoints[k] = {}
                self.checkpoints[k][s] = copy.deepcopy(model.state_dict())
                # Skip to next slice
                continue
            # --------------------------------

            loader = self._get_loader(self.shards[k][s])

            model.train()
            for epoch in range(CONFIG["EPOCHS_PER_SLICE"]):
                for imgs, labels in loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(model(imgs), labels)
                    loss.backward()
                    optimizer.step()

            # 3. Save Checkpoint
            if k not in self.checkpoints:
                self.checkpoints[k] = {}
            self.checkpoints[k][s] = copy.deepcopy(model.state_dict())

        # 4. Update Final Head
        self.shard_models[k] = copy.deepcopy(model.state_dict())

    def train_full_pipeline(self):
        start_time = time.time()
        for k in range(CONFIG["K_SHARDS"]):
            self.train_shard_incremental(k, 0)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        return time.time() - start_time

    def predict(self, indices):
        loader = DataLoader(
            Subset(self.test_set, indices), batch_size=100, shuffle=False
        )
        final_probs = np.zeros((len(indices), 10))
        num_models = len(self.shard_models)

        with torch.no_grad():
            for k, state in self.shard_models.items():
                model = SimpleCNN().to(DEVICE)
                model.load_state_dict(state)
                model.eval()

                shard_probs = []
                for imgs, _ in loader:
                    imgs = imgs.to(DEVICE)
                    outputs = model(imgs)
                    probs = torch.softmax(outputs, dim=1)
                    shard_probs.append(probs.cpu().numpy())

                final_probs += np.concatenate(shard_probs)

        final_probs /= num_models
        return np.argmax(final_probs, axis=1)

    def unlearn(self, delete_indices):
        start_time = time.time()
        impact_map = {}  # {shard_id: min_slice_index}

        # 1. Logical Deletion
        for idx in delete_indices:
            if idx not in self.indices_map:
                continue
            k, s = self.indices_map[idx]

            # Remove index from data structure
            current_indices = self.shards[k][s]
            self.shards[k][s] = current_indices[current_indices != idx]
            del self.indices_map[idx]

            # Track earliest impact point
            if k not in impact_map:
                impact_map[k] = s
            else:
                impact_map[k] = min(impact_map[k], s)

        # 2. Selective Retraining
        slices_retrained = 0
        for k, start_s in impact_map.items():
            self.train_shard_incremental(k, start_s)
            slices_retrained += CONFIG["S_SLICES"] - start_s

        if DEVICE.type == "mps":
            torch.mps.synchronize()
        return time.time() - start_time, slices_retrained

    def get_confidence_scores(self, dataset):
        loader = DataLoader(dataset, batch_size=100, shuffle=False)
        confs = []
        num_models = len(self.shard_models)

        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                batch_probs = torch.zeros(imgs.size(0), 10).to(DEVICE)

                for k, state in self.shard_models.items():
                    m = SimpleCNN().to(DEVICE)
                    m.load_state_dict(state)
                    m.eval()
                    batch_probs += torch.softmax(m(imgs), dim=1)

                batch_probs /= num_models
                max_conf, _ = torch.max(batch_probs, dim=1)
                confs.extend(max_conf.cpu().numpy())
        return np.array(confs)


def select_deletion_indices(sisa: SISAEngine, pct: float, mode: str = "recent"):
    n_delete = int(len(sisa.train_set) * pct)

    available_indices = list(sisa.indices_map.keys())

    if mode == "random":
        return np.random.choice(available_indices, n_delete, replace=False)

    if mode == "recent":
        chosen = []
        remaining = n_delete

        for s in reversed(range(CONFIG["S_SLICES"])):
            for k in range(CONFIG["K_SHARDS"]):

                slice_indices = sisa.shards[k][s]

                candidates = [idx for idx in slice_indices if idx in sisa.indices_map]

                if remaining <= 0:
                    break

                if len(candidates) <= remaining:
                    chosen.extend(candidates)
                    remaining -= len(candidates)
                else:
                    chosen.extend(
                        np.random.choice(candidates, remaining, replace=False).tolist()
                    )
                    remaining = 0
                    break

            if remaining <= 0:
                break

        return np.array(chosen)

    raise ValueError(f"Unknown DELETION_MODE: {mode}")


def run_assignment():
    set_seed(CONFIG["SEED"])

    # transform = transforms.Compose(
    #     [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    # )
    # train_full = torchvision.datasets.CIFAR10(
    #     root="./data", train=True, download=True, transform=transform
    # )
    # test_full = torchvision.datasets.CIFAR10(
    #     root="./data", train=False, download=True, transform=transform
    # )

    # sisa = SISAEngine(train_full, test_full)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_full = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    sisa = SISAEngine(train_full, test_full)

    print("\n--- Phase A: Baseline Training ---")
    mem_start = get_memory_mb()
    base_time = sisa.train_full_pipeline()
    mem_end = get_memory_mb()

    # Baseline Metrics
    y_pred = sisa.predict(range(len(test_full)))
    base_acc = accuracy_score(test_full.targets, y_pred)
    base_f1 = f1_score(test_full.targets, y_pred, average="macro")

    print("   -> Computing Global Baseline MIA...")
    subset_size = 1000
    train_subset = Subset(
        train_full, np.random.choice(len(train_full), subset_size, replace=False)
    )
    test_subset = Subset(
        test_full, np.random.choice(len(test_full), subset_size, replace=False)
    )

    base_train_conf = sisa.get_confidence_scores(train_subset)
    base_test_conf = sisa.get_confidence_scores(test_subset)

    y_true_base = [1] * len(base_train_conf) + [0] * len(base_test_conf)
    y_scores_base = np.concatenate([base_train_conf, base_test_conf])
    global_mia_auc = roc_auc_score(y_true_base, y_scores_base)

    print(
        f"Baseline: Time={base_time:.1f}s | Acc={base_acc:.4f} | F1={base_f1:.4f} | Global MIA={global_mia_auc:.4f}"
    )
    print(f"Memory Delta: ~{mem_end - mem_start:.1f} MB")

    sisa.backup_state()

    results_log = []

    for pct in CONFIG["UNLEARN_PCTS"]:
        print(f"\n--- Experiment: Deleting {pct*100}% ---")
        sisa.restore_state()  # Reset to Baseline

        # Select Targets
        # n_delete = int(len(train_full) * pct)
        # available_indices = list(sisa.indices_map.keys())
        # target_indices = np.random.choice(available_indices, n_delete, replace=False)
        # Select Targets (respecting deletion mode)
        target_indices = select_deletion_indices(
            sisa, pct, mode=CONFIG.get("DELETION_MODE", "recent")
        )

        target_subset = Subset(train_full, target_indices)
        naive_target_conf = sisa.get_confidence_scores(target_subset)
        naive_test_conf = sisa.get_confidence_scores(
            test_full
        )  # Full test set for stability

        y_true_naive = [1] * len(naive_target_conf) + [0] * len(naive_test_conf)
        y_scores_naive = np.concatenate([naive_target_conf, naive_test_conf])
        mia_auc_naive = roc_auc_score(y_true_naive, y_scores_naive)

        t_unlearn, slices_hit = sisa.unlearn(target_indices)

        y_pred_new = sisa.predict(range(len(test_full)))
        acc_new = accuracy_score(test_full.targets, y_pred_new)
        f1_new = f1_score(test_full.targets, y_pred_new, average="macro")

        sisa_target_conf = sisa.get_confidence_scores(target_subset)
        sisa_test_conf = sisa.get_confidence_scores(test_full)

        y_true_sisa = [1] * len(sisa_target_conf) + [0] * len(sisa_test_conf)
        y_scores_sisa = np.concatenate([sisa_target_conf, sisa_test_conf])
        mia_auc_sisa = roc_auc_score(y_true_sisa, y_scores_sisa)

        # Derived Metrics
        time_saved_pct = (1 - (t_unlearn / base_time)) * 100
        frac_slices = slices_hit / (CONFIG["K_SHARDS"] * CONFIG["S_SLICES"])

        print(f"Unlearn Time: {t_unlearn:.1f}s (Saved {time_saved_pct:.1f}%)")
        print(f"Acc: {base_acc:.4f} -> {acc_new:.4f}")
        print(f"MIA AUC: Naïve={mia_auc_naive:.4f} -> SISA={mia_auc_sisa:.4f}")

        results_log.append(
            {
                "pct_deleted": pct * 100,
                "k": CONFIG["K_SHARDS"],
                "s": CONFIG["S_SLICES"],
                "baseline_time": base_time,
                "unlearn_time": t_unlearn,
                "time_saved_pct": time_saved_pct,
                "slices_retrained": slices_hit,
                "frac_slices_retrained": frac_slices,
                "acc_before": base_acc,
                "acc_after": acc_new,
                "f1_after": f1_new,
                "mia_auc_naive": mia_auc_naive,
                "mia_auc_sisa": mia_auc_sisa,
            }
        )

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results_log)
    df.to_csv("results/table1.csv", index=False)

    # Fig 1: Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(df["pct_deleted"], df["acc_after"], "g-o", label="SISA Acc")
    plt.axhline(base_acc, color="r", linestyle="--", label="Baseline")
    plt.xlabel("% Data Deleted")
    plt.ylabel("Test Accuracy")
    plt.title("Fig 1: Model Utility after Unlearning")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/fig1_acc.png")

    # Fig 2: Time Saved
    plt.figure(figsize=(6, 4))
    plt.plot(df["pct_deleted"], df["time_saved_pct"], "b-s")
    plt.xlabel("% Data Deleted")
    plt.ylabel("Time Saved (%)")
    plt.title("Fig 2: Computational Efficiency")
    plt.ylim(0, 105)
    plt.grid(True)
    plt.savefig("results/fig2_time.png")

    # Fig 3: Privacy (MIA)
    plt.figure(figsize=(6, 4))
    x = np.arange(len(df["pct_deleted"]))
    w = 0.35
    plt.bar(
        x - w / 2, df["mia_auc_naive"], w, label="Naïve (No Retrain)", color="salmon"
    )
    plt.bar(x + w / 2, df["mia_auc_sisa"], w, label="SISA (Unlearned)", color="skyblue")
    plt.xticks(x, df["pct_deleted"])
    plt.xlabel("% Data Deleted")
    plt.ylabel("MIA AUC")
    plt.title("Fig 3: Privacy Risk Reduction")
    plt.axhline(0.5, color="k", linestyle="--", label="Ideal Privacy (Random)")
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("results/fig3_mia.png")

    print("\nDone. Results saved to /results.")


if __name__ == "__main__":
    run_assignment()
