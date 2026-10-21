#!/usr/bin/env python3
"""
Evaluate BHCR Sudoku Solver on two popular Kaggle datasets.

Datasets:
1. 9 Million Sudoku Puzzles (rohanrao/sudoku)
2. 3 Million Sudoku Puzzles with Ratings (radcliffe/3-million-sudoku-puzzles-with-ratings)

Requirements:
    pip install kaggle pandas torch numpy tqdm

Setup Kaggle API:
    1. Go to https://www.kaggle.com/settings → Create New Token
    2. Place kaggle.json in ~/.kaggle/kaggle.json
    3. chmod 600 ~/.kaggle/kaggle.json
"""

import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ====================== YOUR MODEL IMPORTS ======================
from bhcr_architecture import BHCRSudokuSolver, neural_guided_bitmask_finish, is_board_valid
from dataset_pipeline import create_initial_candidate_mask
# ================================================================


# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "bhcr_sudoku_solver.pt"

# How many puzzles to evaluate from each dataset (set higher if you have time/GPU)
SAMPLE_SIZE_9M = 100000          # from the 9M dataset
SAMPLE_SIZE_3M = 100000          # from the 3M dataset
ONLY_HARD_PUZZLES = True       # for 3M dataset: keep only difficulty >= 3.0
SEED = 42

DATA_DIR = Path("./sudoku_datasets")
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------
# Model loading (once)
# ---------------------------------------------------------------
def load_model():
    print(f"Using device: {DEVICE}")
    model = BHCRSudokuSolver(hidden_dim=96, max_iterations=10).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully!\n")
    return model


# ---------------------------------------------------------------
# Core solve function (adapted from your sample)
# ---------------------------------------------------------------
def solve_string_puzzle(model, puzzle_str: str):
    clean_str = puzzle_str.strip().replace(' ', '').replace('\n', '').replace('.', '0')
    if len(clean_str) != 81:
        raise ValueError(f"Puzzle must be exactly 81 characters. Got {len(clean_str)}")

    quiz = np.fromiter(
        (int(ch) if ch in '123456789' else 0 for ch in clean_str),
        dtype=np.int64, count=81
    ).reshape(9, 9)

    mask = create_initial_candidate_mask(quiz)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
        step_logits = model(mask_tensor, early_exit_inference=True)
        final_logits = step_logits[-1].float().cpu().numpy()[0]

    solved_zero_idx = neural_guided_bitmask_finish(quiz, final_logits)
    elapsed_ms = (time.perf_counter() - start) * 1000

    solved = solved_zero_idx + 1  # 1-9
    is_valid = is_board_valid(solved_zero_idx)

    return solved, is_valid, elapsed_ms


# ---------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------
def evaluate_on_puzzles(model, puzzles, solutions, dataset_name: str):
    print(f"\n{'='*60}")
    print(f"Evaluating on: {dataset_name}")
    print(f"Number of puzzles: {len(puzzles)}")
    print(f"{'='*60}")

    exact_matches = 0
    cell_correct = 0
    total_cells = 0
    times = []
    failures = []

    for i, (puz, true_sol) in enumerate(tqdm(zip(puzzles, solutions), total=len(puzzles))):
        try:
            pred, is_valid, ms = solve_string_puzzle(model, puz)
            times.append(ms)

            pred_str = ''.join(map(str, pred.flatten()))
            true_str = true_sol.strip().replace(' ', '')

            if pred_str == true_str and is_valid:
                exact_matches += 1
            else:
                failures.append(i)

            # Cell-level accuracy
            pred_flat = pred.flatten()
            true_flat = np.array([int(c) for c in true_str])
            cell_correct += np.sum(pred_flat == true_flat)
            total_cells += 81

        except Exception as e:
            print(f"\nError on puzzle {i}: {e}")
            failures.append(i)

    exact_acc = exact_matches / len(puzzles) * 100
    cell_acc = cell_correct / total_cells * 100
    avg_time = np.mean(times) if times else 0
    p95_time = np.percentile(times, 95) if times else 0

    print(f"\nResults for {dataset_name}:")
    print(f"  Exact Match Accuracy : {exact_acc:.2f}%  ({exact_matches}/{len(puzzles)})")
    print(f"  Cell Accuracy        : {cell_acc:.2f}%")
    print(f"  Avg solve time       : {avg_time:.2f} ms")
    print(f"  95th percentile time : {p95_time:.2f} ms")
    print(f"  Failed / Exception   : {len(failures)}")

    return {
        "dataset": dataset_name,
        "exact_match": exact_acc,
        "cell_accuracy": cell_acc,
        "avg_ms": avg_time,
        "p95_ms": p95_time,
        "n": len(puzzles),
        "failures": len(failures)
    }


# ---------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------
def download_kaggle_dataset(dataset_slug: str, target_dir: Path):
    """Download and unzip a Kaggle dataset using the official API."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dataset_slug} ...")

    # Using kaggle CLI (most reliable)
    cmd = f'kaggle datasets download -d {dataset_slug} -p {target_dir} --unzip'
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(
            f"Failed to download {dataset_slug}.\n"
            "Make sure you have:\n"
            "  1. pip install kaggle\n"
            "  2. ~/.kaggle/kaggle.json with your API token\n"
            "  3. Accepted the dataset rules on the Kaggle website"
        )
    print(f"Downloaded to {target_dir}")


# ---------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------
def load_9m_sample(n_samples=2000):
    """Load a random sample from the 9 Million dataset (robust column detection)."""
    data_path = DATA_DIR / "9m"
    csv_path = data_path / "sudoku.csv"

    if not csv_path.exists():
        download_kaggle_dataset("rohanrao/sudoku", data_path)

    print(f"Loading sample of {n_samples} puzzles from 9M dataset...")

    # Inspect columns safely using string type
    df_sample = pd.read_csv(csv_path, nrows=5, dtype=str)
    print("Columns found:", list(df_sample.columns))

    # Detect puzzle & solution columns robustly
    cols = [c.lower() for c in df_sample.columns]

    if "quizzes" in cols or "puzzle" in cols or "quiz" in cols:
        puzzle_col = next(c for c in df_sample.columns if c.lower() in ("quizzes", "puzzle", "quiz"))
        solution_col = next(c for c in df_sample.columns if c.lower() in ("solutions", "solution", "sol"))
    else:
        # Fallback to first two columns
        puzzle_col = df_sample.columns[0]
        solution_col = df_sample.columns[1]

    print(f"Using columns → Puzzle: '{puzzle_col}' | Solution: '{solution_col}'")

    # Load the data, strictly forcing string type to preserve all 81 digits
    df = pd.read_csv(csv_path, usecols=[puzzle_col, solution_col], dtype=str)
    
    # Sample the requested number of puzzles
    df = df.sample(n=min(n_samples, len(df)), random_state=SEED)

    puzzles = df[puzzle_col].tolist()
    solutions = df[solution_col].tolist()

    return puzzles, solutions



def load_3m_sample(n_samples=2000, only_hard=True):
    """Load a sample from the 3 Million dataset (with difficulty ratings)."""
    data_path = DATA_DIR / "3m"
    # The file is usually named something like "sudoku.csv" or similar
    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        download_kaggle_dataset("radcliffe/3-million-sudoku-puzzles-with-ratings", data_path)
        csv_files = list(data_path.glob("*.csv"))

    csv_path = csv_files[0]
    print(f"Loading from {csv_path.name} ...")

    df = pd.read_csv(csv_path)

    # Common column names in this dataset
    # Usually: puzzle, solution, clues, difficulty (or similar)
    puzzle_col = next((c for c in df.columns if "puzzle" in c.lower() or "quiz" in c.lower()), df.columns[0])
    solution_col = next((c for c in df.columns if "solution" in c.lower() or "sol" in c.lower()), df.columns[1])
    diff_col = next((c for c in df.columns if "diff" in c.lower() or "rating" in c.lower()), None)

    if only_hard and diff_col is not None:
        print(f"Filtering hard puzzles (difficulty >= 3.0) using column '{diff_col}'")
        df = df[df[diff_col] >= 3.0]

    df = df.sample(n=min(n_samples, len(df)), random_state=SEED)

    puzzles = df[puzzle_col].astype(str).tolist()
    solutions = df[solution_col].astype(str).tolist()

    return puzzles, solutions


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    model = load_model()

    results = []

    # ---------- Dataset 1: 9 Million ----------
    try:
        puzzles_9m, sols_9m = load_9m_sample(SAMPLE_SIZE_9M)
        res1 = evaluate_on_puzzles(model, puzzles_9m, sols_9m, "9M Sudoku (rohanrao)")
        results.append(res1)
    except Exception as e:
        print(f"\nFailed on 9M dataset: {e}")

    # ---------- Dataset 2: 3 Million with ratings ----------
    try:
        puzzles_3m, sols_3m = load_3m_sample(SAMPLE_SIZE_3M, only_hard=ONLY_HARD_PUZZLES)
        res2 = evaluate_on_puzzles(
            model, puzzles_3m, sols_3m,
            "3M Sudoku with Ratings (hard subset)" if ONLY_HARD_PUZZLES else "3M Sudoku with Ratings"
        )
        results.append(res2)
    except Exception as e:
        print(f"\nFailed on 3M dataset: {e}")

    # ---------- Summary ----------
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    for r in results:
        print(f"{r['dataset']:<45} Exact: {r['exact_match']:6.2f}%  |  "
              f"Cell: {r['cell_accuracy']:6.2f}%  |  Avg: {r['avg_ms']:.1f} ms")


if __name__ == "__main__":
    main()