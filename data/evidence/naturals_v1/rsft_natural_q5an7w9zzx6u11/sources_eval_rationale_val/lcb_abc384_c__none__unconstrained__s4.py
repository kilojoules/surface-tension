import sys
from itertools import product

def solve():
    # Read input scores
    scores_input = list(map(int, sys.stdin.read().split()))
    if not scores_input:
        return
    
    # Problem names and their corresponding scores
    problems = ['A', 'B', 'C', 'D', 'E']
    
    # Generate all 32 combinations of solving/not solving problems (0 or 1)
    # Use a list comprehension to create participants and their total scores
    # Each participant is represented as a string of solved problems
    # We filter out the case where no problems are solved (the empty string)
    participants = [
        (
            "".join([problems[i] for i in range(5) if solved[i]]),
            sum([scores_input[i] for i in range(5) if solved[i]])
        )
        for solved in product([0, 1], repeat=5)
        if any(solved)
    ]
    
    # Sort the participants:
    # 1. By score descending (-x[1])
    # 2. By name ascending (x[0])
    sorted_participants = sorted(participants, key=lambda x: (-x[1], x[0]))
    
    # Print the names using a join and map to avoid explicit loops
    sys.stdout.write("\n".join(map(lambda x: x[0], sorted_participants)) + "\n")

if __name__ == "__main__":
    solve()