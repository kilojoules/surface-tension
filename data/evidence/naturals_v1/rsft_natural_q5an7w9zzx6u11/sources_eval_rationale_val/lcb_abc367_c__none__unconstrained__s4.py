import sys
from itertools import product

def solve():
    # Read N and K from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    # Read R_i values
    R = [int(x) for x in input_data[2:]]
    
    # Generate ranges for each position: (1, 2, ..., R_i)
    # We use range(1, r + 1) for each R_i in R
    ranges = [range(1, r + 1) for r in R]
    
    # itertools.product generates the Cartesian product of the input iterables.
    # It generates them in lexicographical order based on the order of the input iterables.
    # We use a list comprehension to filter sequences where the sum is a multiple of K.
    results = [
        " ".join(map(str, seq)) 
        for seq in product(*ranges) 
        if sum(seq) % K == 0
    ]
    
    # Print all valid sequences joined by newlines
    sys.stdout.write("\n".join(results) + ("\n" if results else ""))

if __name__ == "__main__":
    solve()