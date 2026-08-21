import sys
from itertools import product

def solve():
    # Read N and K from the first line
    # Read R_1...R_N from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    R = [int(x) for x in input_data[2:]]
    
    # Generate ranges for each position: (1, 2, ..., R_i)
    # itertools.product generates sequences in lexicographical order 
    # if the input iterables are sorted.
    ranges = [range(1, r + 1) for r in R]
    
    # Use a list comprehension to filter sequences where sum % K == 0
    # This replaces the need for a for-loop and if-statement block.
    results = [
        " ".join(map(str, seq)) 
        for seq in product(*ranges) 
        if sum(seq) % K == 0
    ]
    
    # Print all results joined by newlines
    sys.stdout.write("\n".join(results) + ("\n" if results else ""))

if __name__ == "__main__":
    solve()