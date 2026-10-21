import torch
import numpy as np
import time
from torch.utils.data import DataLoader
from bhcr_architecture import BHCRSudokuSolver, neural_guided_bitmask_finish
from dataset_pipeline import AutomaticSudokuBenchmark, create_initial_candidate_mask

def evaluate(model, loader, device, max_puzzles=None):
    model.eval()
    total = 0
    pure_neural_exact = 0
    hybrid_exact = 0
    pure_neural_cells = 0
    total_cells = 0

    # Warmup
    sample = next(iter(loader))
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=device.type=='cuda'):
        for _ in range(3):
            _ = model(sample[1][:8].to(device), early_exit_inference=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.no_grad():
        for quizzes, masks, solutions in loader:
            if max_puzzles and total >= max_puzzles:
                break

            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=device.type=='cuda'):
                logits = model(masks, early_exit_inference=True)[-1].float().cpu().numpy()

            quizzes_np = quizzes.numpy()
            sols_np = solutions.numpy()

            neural_preds = np.argmax(logits, axis=-1)
            cell_correct = (neural_preds == sols_np)
            pure_neural_cells += cell_correct.sum()
            total_cells += sols_np.size

            board_correct = cell_correct.reshape(len(quizzes_np), 81).all(axis=1)
            pure_neural_exact += board_correct.sum()

            for i in range(len(quizzes_np)):
                if board_correct[i]:
                    hybrid_exact += 1
                else:
                    finished = neural_guided_bitmask_finish(quizzes_np[i], logits[i])
                    if np.array_equal(finished, sols_np[i]):
                        hybrid_exact += 1

            total += len(quizzes_np)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return {
        "total": total,
        "pure_neural_board": pure_neural_exact / total * 100,
        "hybrid_board": hybrid_exact / total * 100,
        "pure_neural_cell": pure_neural_cells / total_cells * 100,
        "ms_per_puzzle": (elapsed * 1000) / total
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = BHCRSudokuSolver(hidden_dim=96, max_iterations=10).to(device)
    model.load_state_dict(torch.load("bhcr_sudoku_solver.pt", map_location=device))
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Evaluate on hard test set
    print("\nLoading hard test set...")
    test_ds = AutomaticSudokuBenchmark(split="test", max_samples=5000, augment_train=False)  # increase if you want
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, pin_memory=True)

    print("Running evaluation...")
    results = evaluate(model, test_loader, device)

    print("\n" + "="*60)
    print("BHCR-Net Evaluation Results")
    print("="*60)
    print(f"Puzzles tested              : {results['total']:,}")
    print(f"Pure Neural Board Accuracy  : {results['pure_neural_board']:.2f}%")
    print(f"Hybrid Board Accuracy       : {results['hybrid_board']:.2f}%")
    print(f"Pure Neural Cell Accuracy   : {results['pure_neural_cell']:.2f}%")
    print(f"Speed                       : {results['ms_per_puzzle']:.3f} ms/puzzle")
    print("="*60)