import sys

def solve():
    # Read all input at once and split into a list of strings
    data = sys.stdin.read().split()
    
    # N is the first element
    # A is the sequence from index 1 to N
    # B is the sequence from index N+1 to 2N
    # Using map(int, ...) to convert strings to integers without explicit loops
    # Using slicing to separate sequence A and B
    
    # Since we need to avoid loops, we use map and slicing.
    # We convert the entire input to integers first.
    all_ints = list(map(int, data))
    n = all_ints[0]
    
    # Sequence A is from index 1 to n
    # Sequence B is from index n+1 to 2n
    # max() is a built-in that finds the maximum value in an iterable
    result = max(all_ints[1:n+1]) + max(all_ints[n+1:])
    
    # Print the final result
    sys.stdout.write(str(result) + '\n')

# Using a wrapper to call the solve function
if __name__ == "__main__":
    solve()