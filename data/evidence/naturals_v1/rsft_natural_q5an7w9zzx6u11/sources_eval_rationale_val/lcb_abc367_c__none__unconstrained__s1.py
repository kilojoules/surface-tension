import itertools
import sys

def solve():
    # Read N and K from the first line
    # Read R_i values from the second line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    K = int(input_data[1])
    R = [int(x) for x in input_data[2:]]

    # Generate ranges for each R_i: (1, 2, ..., R_i)
    ranges = [range(1, r + 1) for r in R]

    # itertools.product generates the Cartesian product of the ranges.
    # Because the ranges are sorted and product iterates through them 
    # in order, the resulting sequences are in lexicographical order.
    # We use a list comprehension to filter sequences where sum(seq) % K == 0.
    results = [
        " ".join(map(str, seq))
        for seq in itertools.product(*ranges)
        if sum(seq) % K == 0
    ]

    # Print all valid sequences joined by newlines
    sys.stdout.write("\n".join(results) + ("\n" if results else ""))

if __name__ == "__main__":
    solve()