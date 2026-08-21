import itertools
import sys

def solve():
    # Read N and K from the first line of input
    # Read R_1...R_N from the second line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    R = [int(x) for x in input_data[2:]]

    # Generate ranges for each position: (1, 2, ..., R_i)
    # itertools.product generates the Cartesian product of the input iterables.
    # It produces sequences in lexicographical order if the input iterables are sorted.
    ranges = [range(1, r + 1) for r in R]
    
    # Use a list comprehension to filter sequences where the sum is a multiple of K.
    # This replaces the need for a for-loop and an if-statement.
    results = [
        " ".join(map(str, seq)) 
        for seq in itertools.product(*ranges) 
        if sum(seq) % K == 0
    ]
    
    # Print all valid sequences joined by newlines.
    # Using sys.stdout.write to avoid a loop for printing.
    sys.stdout.write("\n".join(results) + ("\n" if results else ""))

if __name__ == "__main__":
    solve()