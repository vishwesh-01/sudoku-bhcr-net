import torch
import numpy as np
from bhcr_architecture import BHCRSudokuSolver, neural_guided_bitmask_finish, is_board_valid
from dataset_pipeline import create_initial_candidate_mask

def solve_string_puzzle(puzzle_str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load trained model weights
    model = BHCRSudokuSolver(hidden_dim=96, max_iterations=10).to(device)
    model.load_state_dict(torch.load("bhcr_sudoku_solver.pt", map_location=device))
    model.eval()

    # Clean string input (accepts '.' or '0' for empty cells)
    clean_str = puzzle_str.strip().replace(' ', '').replace('\n', '')
    if len(clean_str) != 81:
        raise ValueError(f"Puzzle string must be exactly 81 characters! Got {len(clean_str)}.")

    quiz = np.fromiter((int(ch) if ch in '123456789' else 0 for ch in clean_str), dtype=np.int64, count=81).reshape(9, 9)
    mask = create_initial_candidate_mask(quiz)
    
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device)

    # Run inference
    import time
    start_t = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
        step_logits = model(mask_tensor, early_exit_inference=True)
        final_logits = step_logits[-1].float().cpu().numpy()[0]
    
    solved_zero_idx = neural_guided_bitmask_finish(quiz, final_logits)
    elapsed_ms = (time.perf_counter() - start_t) * 1000.0

    solved_1_indexed = solved_zero_idx + 1
    if is_board_valid(solved_zero_idx):
        print(f"\n[Success] Puzzle solved successfully in {elapsed_ms:.2f} ms!")
    else:
        print(f"\n[Warning] Board verification failed.")

    # Print nicely formatted grid
    print("\nSolved Board:")
    for r in range(9):
        if r % 3 == 0 and r != 0:
            print("-" * 21)
        row_str = ""
        for c in range(9):
            if c % 3 == 0 and c != 0:
                row_str += "| "
            row_str += f"{solved_1_indexed[r, c]} "
        print(row_str)

if __name__ == "__main__":
    # Example: World's Hardest Sudoku by Arto Inkala
    inkala_hardest = (
        "8.................."
        ".73.8......"
        "..9...6...."
        ".5..4onta...." # (or standard string format)
    )
    # Let's use a standard extreme string:
    sample_extreme = "1........ .0.2.... .3.5.#..." # or any 81 char string
    # Let's run with a clean 81-char string example:
    test_puzzle = "800000000003600000070090200000007000040000100020000000008040020000001760000003004"
    
    print("Solving custom input puzzle...")
    solve_string_puzzle(test_puzzle)