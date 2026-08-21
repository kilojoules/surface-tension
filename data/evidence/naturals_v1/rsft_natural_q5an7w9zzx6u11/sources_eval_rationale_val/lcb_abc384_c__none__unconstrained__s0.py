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
    
    # Generate all 31 non-empty subsequences of "ABCDE"
    # product([0, 1], repeat=5) creates all binary combinations for A, B, C, D, E
    participants = [
        "".join([prob for prob, solved in zip("ABCDE", combo) if solved])
        for combo in product([0, 1], repeat=5)
        if any(combo)
    ]
    
    # Calculate score for a participant name
    # Score is the sum of scores of problems they solved
    get_score = lambda name: sum(scores_map[char] for char in name)
    
    # Sort participants:
    # 1. By score descending (-get_score(p))
    # 2. By name lexicographically ascending (p)
    sorted_participants = sorted(
        participants, 
        key=lambda p: (-get_score(p), p)
    )
    
    # Output the result
    sys.stdout.write("\n".join(sorted_participants) + "\n")

if __name__ == "__main__":
    solve()