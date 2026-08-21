import sys
from itertools import product

def solve():
    # Read input scores
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map problem letters to their respective scores
    scores_map = dict(zip("ABCDE", map(int, input_data)))
    
    # Generate all 31 non-empty combinations of problems solved
    # product([0, 1], repeat=5) creates all binary strings of length 5
    # We filter out the case where no problems are solved (all 0s)
    participants = [
        "".join([prob for prob, solved in zip("ABCDE", combo) if solved])
        for combo in product([0, 1], repeat=5)
        if any(combo)
    ]
    
    # Function to calculate the total score for a participant name
    # Using sum() with a generator expression inside
    get_score = lambda name: sum(scores_map[char] for char in name)
    
    # Sort the participants:
    # 1. By score descending (-get_score(p))
    # 2. By name ascending (p)
    sorted_participants = sorted(
        participants, 
        key=lambda p: (-get_score(p), p)
    )
    
    # Output the result
    sys.stdout.write("\n".join(sorted_participants) + "\n")

if __name__ == "__main__":
    solve()