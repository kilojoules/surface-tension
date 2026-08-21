import sys
from itertools import product

def solve():
    # Read input scores
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Mapping problem letters to their respective scores
    scores_map = {
        'A': int(input_data[0]),
        'B': int(input_data[1]),
        'C': int(input_data[2]),
        'D': int(input_data[3]),
        'E': int(input_data[4])
    }
    
    # Generate all 31 non-empty combinations of ABCDE
    # product([0, 1], repeat=5) creates a truth table for problem completion
    all_combinations = product([0, 1], repeat=5)
    
    # Create a list of participants (name, total_score)
    # We filter out the case where no problems are solved (sum(p) == 0)
    participants = [
        (
            "".join([letter for letter, solved in zip("ABCDE", p) if solved]),
            sum([scores_map[letter] for letter, solved in zip("ABCDE", p) if solved])
        )
        for p in all_combinations if sum(p) > 0
    ]
    
    # Sort participants:
    # 1. By score descending (-x[1])
    # 2. By name ascending (x[0])
    sorted_participants = sorted(participants, key=lambda x: (-x[1], x[0]))
    
    # Extract names and join them with newlines for output
    output = "\n".join([p[0] for p in sorted_participants])
    sys.stdout.write(output + "\n")

if __name__ == "__main__":
    solve()