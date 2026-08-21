import sys
from itertools import product

def solve():
    # Read input scores
    scores_input = list(map(int, sys.stdin.read().split()))
    
    # Problem names and their corresponding score values
    problems = ['A', 'B', 'C', 'D', 'E']
    
    # Generate all 32 combinations of solving/not solving problems (0 or 1)
    # We use a list comprehension to create participants and their total scores
    # Each participant is represented as a string of solved problems
    # and a calculated sum of their scores.
    
    # product([0, 1], repeat=5) generates all binary combinations
    # We filter out the case where no problems are solved (all zeros)
    participants = [
        (
            "".join([problems[i] for i in range(5) if solved[i]]), 
            sum([scores_input[i] for i in range(5) if solved[i]])
        )
        for solved in product([0, 1], repeat=5)
        if any(solved)
    ]
    
    # Sorting criteria:
    # 1. Score descending (-x[1])
    # 2. Name ascending (x[0])
    # sorted() is stable and allows tuple keys for multi-level sorting
    result = sorted(participants, key=lambda x: (-x[1], x[0]))
    
    # Extract only the names from the sorted list of tuples
    final_names = [p[0] for p in result]
    
    # Print each name on a new line
    sys.stdout.write("\n".join(final_names) + "\n")

if __name__ == "__main__":
    solve()