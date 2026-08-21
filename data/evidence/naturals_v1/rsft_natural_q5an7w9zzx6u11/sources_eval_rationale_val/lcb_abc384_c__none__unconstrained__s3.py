import itertools

def solve():
    # Read input scores
    scores_input = list(map(int, input().split()))
    problem_names = "ABCDE"
    
    # Generate all 31 non-empty combinations of problems solved
    # itertools.product([0, 1], repeat=5) creates all binary strings of length 5
    # We use a list comprehension to create tuples of (name, total_score)
    participants = [
        (
            "".join([problem_names[i] for i in range(5) if solved[i]]),
            sum([scores_input[i] for i in range(5) if solved[i]])
        )
        for solved in itertools.product([0, 1], repeat=5)
        if any(solved) # Ensure the participant solved at least one problem
    ]
    
    # Sort the participants:
    # 1. By score descending (-x[1])
    # 2. By name ascending (x[0])
    sorted_participants = sorted(participants, key=lambda x: (-x[1], x[0]))
    
    # Print the names using a join and map to avoid explicit loops
    print("\n".join(map(lambda x: x[0], sorted_participants)))

if __name__ == "__main__":
    solve()