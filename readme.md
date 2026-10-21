# BHCR-Net: Bipartite House-Cell Reasoner for Sudoku

**A hybrid neural-symbolic Sudoku solver that achieves 100% exact accuracy on extreme and large-scale benchmarks while remaining extremely fast.**

BHCR-Net combines a lightweight recurrent neural architecture that simultaneously reasons from both **cell-centric** and **house-centric** perspectives with a neural-guided bitmask backtracker. The result is a solver that is both highly accurate and practical for real-world use.

---

## Key Results

| Dataset | Puzzles | Exact Board Accuracy | Cell Accuracy | Avg. Time | Notes |
|---------|---------|----------------------|---------------|-----------|-------|
| **sapientinc/sudoku-extreme-1k** (test_hard) | 2,000 | **100.00%** (hybrid) | ~68.7% (pure neural) | ~38 ms | Extremely hard puzzles |
| **rohanrao/sudoku** (9M dataset sample) | 100,000 | **100.00%** | **100.00%** | **22.9 ms** | Random sample |
| **radcliffe/3-million-sudoku-puzzles-with-ratings** (hard subset, difficulty ≥ 3.0) | 100,000 | **100.00%** | **100.00%** | **46.6 ms** | Hard-rated puzzles only |

- Pure neural board exact-match on extreme puzzles is intentionally low (~3%) because the network focuses on high-quality probability rankings rather than forcing a complete solution.
- The neural-guided bitmask finisher guarantees **100% exact solutions** in microseconds when needed.
- Early-exit inference + FP16 autocast deliver excellent throughput on modern GPUs (e.g. RTX 4060 Laptop).

---

## Architecture Overview

### Core Innovation: Bipartite House-Cell Layer

At every recurrent step the model maintains four probability views:

1. **Cell-view** – Softmax over digits *within* each cell  
2. **Row-house view** – Softmax over columns for each digit in a row  
3. **Column-house view** – Softmax over rows for each digit in a column  
4. **Box-house view** – Softmax over the 9 cells of each 3×3 box for each digit  

These four views are concatenated, projected, and fused with structural context (row / column / box mean + max pooling) via a gated residual update. This dual perspective (cell ↔ house) is the key inductive bias that allows the network to learn Sudoku logic efficiently.

### Full Pipeline

```
Input Puzzle (9×9)
       │
       ▼
create_initial_candidate_mask()   ← cascades naked singles
       │
       ▼
BHCRSudokuSolver (max 10 recurrent steps, early exit possible)
       │
       ▼
Neural Logits [9×9×9]
       │
       ▼
neural_guided_bitmask_finish()    ← MRV + neural ranking + integer bitmasks
       │
       ▼
Guaranteed Valid Solution
```

- **Hidden dimension**: 96  
- **Parameters**: ~180k  
- **Max iterations**: 10 (with confidence-based early exit)

---

## Project Structure

```
.
├── bhcr_architecture.py      # BHCRSudokuSolver + BipartiteHouseCellLayer + bitmask finisher
├── dataset_pipeline.py       # HuggingFace extreme dataset loader + candidate mask generation
├── train_bhcr.py             # Full training + evaluation loop
├── bhcr_evaluate.py          # Clean evaluation script on the extreme test set
├── evaluate_sodoku_kaggle.py # Large-scale evaluation on 9M and 3M Kaggle datasets
├── solve_custom.py           # Simple CLI / script for solving a single puzzle string
├── sodoku_solver.ipynb       # Interactive Jupyter demo
├── bhcr_sudoku_solver.pt     # Pre-trained weights (~710 KB)
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Create environment (recommended)

```bash
conda create -n bhcr python=3.11
conda activate bhcr
# or
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**CUDA note**: The `requirements.txt` installs the CUDA 12.x / 13.x compatible PyTorch wheel.  
If you have a different CUDA version, visit https://pytorch.org and install the matching torch package first, then the rest of the requirements.

### 3. (Optional) Kaggle API for large-scale evaluation

```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/kaggle.json and run:
chmod 600 ~/.kaggle/kaggle.json
```

---

## Quick Start – Solve a Puzzle

```python
from solve_custom import solve_string_puzzle

# 81-character string ('.' or '0' for empty cells)
puzzle = "800000000003600000070090200000007000040000100020000000008040020000001760000003004"
solve_string_puzzle(puzzle)
```

Or from the command line:

```bash
python solve_custom.py
```

Example output:

```
[Success] Puzzle solved successfully in 39.85 ms!

Solved Board:
8 6 9 | 7 4 2 | 3 1 5 
5 1 2 | 6 8 3 | 7 9 4 
7 3 4 | 5 1 9 | 8 2 6 
---------------------
6 9 1 | 4 3 7 | 2 5 8 
3 4 8 | 2 9 5 | 6 7 1 
2 7 5 | 8 6 1 | 9 4 3 
---------------------
9 5 6 | 3 7 4 | 1 8 2 
4 8 7 | 1 2 6 | 5 3 9 
1 2 3 | 9 5 8 | 4 6 7 
```

---

## Training from Scratch

```bash
python train_bhcr.py
```

- Uses the `sapientinc/sudoku-extreme-1k` dataset (automatically downloaded).
- 8 epochs, AdamW + cosine schedule, deep supervision across recurrent steps.
- Automatic data augmentation (digit permutation + transpose).
- After training the checkpoint `bhcr_sudoku_solver.pt` is saved.

Typical training log (RTX 4060 Laptop):

```
Epoch 01/08 | Loss: 0.4579 | Pure Neural Cell Acc: 64.94% | Pure Neural Board EM: 2.80% | Full BHCR-Net Accuracy: 100.00% | Speed: 40.90 ms/puzzle
...
Epoch 08/08 | Loss: 0.3815 | Pure Neural Cell Acc: 68.70% | Pure Neural Board EM: 3.35% | Full BHCR-Net Accuracy: 100.00% | Speed: 38.97 ms/puzzle
```

---

## Evaluation

### Extreme hard set (HuggingFace)

```bash
python bhcr_evaluate.py
```

### Large-scale Kaggle benchmarks

```bash
# Edit SAMPLE_SIZE_9M / SAMPLE_SIZE_3M inside the script if desired
python evaluate_sodoku_kaggle.py
```

This will download the datasets (if not present), evaluate on 100k puzzles from each, and print exact-match / cell accuracy / latency statistics.

---

## Design Highlights

- **Candidate mask cascading** – Initial naked singles are propagated before the network even starts, giving it a cleaner input.
- **Four-view bipartite reasoning** – Explicitly models both “what digit belongs in this cell” and “which cell holds this digit in the house”.
- **Gated residual update with LayerNorm** – Stabilizes the recurrent process and prevents drift.
- **Neural-guided bitmask finisher** – Uses the network’s ranking as the variable ordering inside a classic exact cover / dancing-links style search with bitmasks. Extremely fast on the few remaining ambiguous cells.
- **Early exit** – Stops as soon as every cell has confidence > 0.96, saving compute on easy boards.
- **FP16 autocast** – Full mixed-precision support for maximum GPU throughput.

---

## Hardware Recommendations

| Hardware              | Expected Performance                  |
|-----------------------|---------------------------------------|
| RTX 4060 / 4070 Laptop | ~23–40 ms per puzzle (batch 1)       |
| RTX 3080 / 4080       | Higher throughput in batched mode    |
| CPU only              | Still works, but significantly slower |

The model is small (~180k parameters) and runs comfortably on consumer GPUs.

---

## Citation / Inspiration

This work builds on classic constraint-propagation ideas and modern neural Sudoku solvers. The bipartite house-cell formulation and the neural-guided integer bitmask finisher are the novel contributions of this repository.

---

## License

This project is released for research and educational purposes.  
Feel free to use, modify, and build upon it.

---

**Enjoy solving the hardest Sudokus at neural speed!** 🧩
