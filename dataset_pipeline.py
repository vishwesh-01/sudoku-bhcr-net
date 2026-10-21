import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download

# Pre-compute peer indices (row, col, and 3x3 box neighbors) for all 81 cells
PEER_INDICES = []
for r in range(9):
    for c in range(9):
        peers = set()
        for i in range(9):
            peers.add((r, i))
            peers.add((i, c))
        br, bc = (r // 3) * 3, (c // 3) * 3
        for dr in range(3):
            for dc in range(3):
                peers.add((br + dr, bc + dc))
        peers.remove((r, c))
        PEER_INDICES.append(list(peers))


def create_initial_candidate_mask(quiz_grid: np.ndarray) -> np.ndarray:
    """
    Turns a 9x9 puzzle into a (9, 9, 9) candidate mask.
    1.0 = digit (1-9) is legally possible in that cell; 0.0 = ruled out.
    Cascades initial naked singles automatically.
    """
    mask = np.ones((9, 9, 9), dtype=np.float32)
    known_queue = []

    for r in range(9):
        for c in range(9):
            val = quiz_grid[r, c]
            if val > 0:
                known_queue.append((r, c, val - 1))

    visited = set()
    while known_queue:
        r, c, d = known_queue.pop(0)
        if (r, c) in visited:
            continue
        visited.add((r, c))

        # Lock this cell to only digit d
        mask[r, c, :] = 0.0
        mask[r, c, d] = 1.0

        # Eliminate digit d from all 20 peers (same row, col, and 3x3 box)
        for pr, pc in PEER_INDICES[r * 9 + c]:
            if mask[pr, pc, d] == 1.0:
                mask[pr, pc, d] = 0.0
                if (pr, pc) not in visited and mask[pr, pc].sum() == 1.0:
                    sole_digit = int(np.argmax(mask[pr, pc]))
                    known_queue.append((pr, pc, sole_digit))

    return mask


class AutomaticSudokuBenchmark(Dataset):
    """
    Downloads and reads 'sapientinc/sudoku-extreme-1k' CSVs directly with dtype=str
    to prevent PyArrow 64-bit integer overflow on 81-digit Sudoku strings.
    """
    def __init__(self, split="train", max_samples=2000, augment_train=True):
        repo_id = "sapientinc/sudoku-extreme-1k"
        filename = "train.csv" if split == "train" else "test_hard.csv"
        print(f"[Dataset] Fetching '{filename}' from {repo_id}...")

        csv_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset"
        )

        # CRITICAL FIX: dtype=str prevents Pandas/PyArrow from converting 81 digits to int64
        df = pd.read_csv(csv_path, dtype=str, nrows=max_samples)
        questions = df["question"].values
        answers = df["answer"].values

        self.augment = (split == "train") and augment_train
        self.quizzes = []
        self.masks = []
        self.sols = []

        print(f"[Dataset] Pre-computing candidate bitmasks for {len(questions):,} puzzles...")
        for q_raw, a_raw in zip(questions, answers):
            q_str = str(q_raw).strip().replace('.', '0')
            a_str = str(a_raw).strip()

            quiz = np.fromiter((int(ch) for ch in q_str), dtype=np.int64, count=81).reshape(9, 9)
            sol = np.fromiter((int(ch) - 1 for ch in a_str), dtype=np.int64, count=81).reshape(9, 9)
            mask = create_initial_candidate_mask(quiz)

            self.quizzes.append(quiz)
            self.masks.append(mask)
            self.sols.append(sol)

        print(f"[Dataset] Ready: {len(self.quizzes):,} extreme puzzles cached in RAM.")

    def __len__(self):
        # Virtually expand the 1,000 training puzzles 10x using symmetry & digit relabeling
        return len(self.quizzes) * (10 if self.augment else 1)

    def __getitem__(self, idx):
        base_idx = idx % len(self.quizzes)
        quiz = self.quizzes[base_idx]
        mask = self.masks[base_idx]
        sol = self.sols[base_idx]

        # Fast vectorized digit-permutation & transpose augmentation during training
        if self.augment and idx >= len(self.quizzes):
            perm = np.random.permutation(9)       # Maps old digit (0..8) -> new digit (0..8)
            inv_perm = np.argsort(perm)           # Reorders candidate channels to match

            sol = perm[sol]
            quiz_aug = np.zeros_like(quiz)
            non_zero = quiz > 0
            quiz_aug[non_zero] = perm[quiz[non_zero] - 1] + 1
            quiz = quiz_aug
            mask = mask[:, :, inv_perm]

            if np.random.rand() > 0.5:
                quiz = quiz.T.copy()
                sol = sol.T.copy()
                mask = np.transpose(mask, (1, 0, 2)).copy()

        return (
            torch.from_numpy(quiz),
            torch.from_numpy(mask),
            torch.from_numpy(sol)
        )


def get_dataloaders(batch_size=128, test_samples=2000):
    train_ds = AutomaticSudokuBenchmark(split="train", max_samples=1000, augment_train=True)
    test_ds = AutomaticSudokuBenchmark(split="test", max_samples=test_samples, augment_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader