import sys
from itertools import product

def solve():
    # Read input scores
    scores = list(map(int, sys.stdin.read().split()))
    problem_names = "ABCDE"
    
    # Generate all 31 non-empty combinations of problems
    # product([0, 1], repeat=5) generates all binary strings of length 5
    # We filter out (0,0,0,0,0) because every participant solved at least one problem
    participants = [
        "".join(problem_names[i] for i in range(5) if solved[i])
        for solved in product([0, 1], repeat=5)
        if any(solved)
    ]
    
    # Function to calculate score for a participant name
    # We use a generator expression inside sum() to avoid explicit loops
    get_score = lambda name: sum(scores[problem_names.index(char)] for char in name)
    
    # Sort participants:
    # 1. By score descending (-get_score(p))
    # 2. By name ascending (p)
    result = sorted(participants, key=lambda p: (-get_score(p), p))
    
    # Print results joined by newlines
    sys.stdout.write("\n".join(result) + "\n")

if __name__ == "__main__":
    solve()