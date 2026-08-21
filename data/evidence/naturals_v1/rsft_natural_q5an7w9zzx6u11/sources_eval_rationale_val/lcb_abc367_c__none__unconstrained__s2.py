import sys
from itertools import product

def solve():
    # Read all input at once and split into a list of integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use a generator to extract N and K, then the remaining R values
    # Since we cannot use loops, we use next() to consume the iterator
    it = input_data
    try:
        n = next(it)
        k = next(it)
        r_values = list(it)
    except StopIteration:
        return

    # Generate ranges for each R_i: (1, 2, ..., R_i)
    # We use a list comprehension to create the range objects
    ranges = [range(1, r + 1) for r in r_values]

    # itertools.product generates the Cartesian product of the ranges.
    # Because the ranges are provided in order and are themselves sorted,
    # the product is generated in lexicographical order by default.
    # We use a list comprehension to filter sequences whose sum is a multiple of K.
    results = [
        seq for seq in product(*ranges) 
        if sum(seq) % k == 0
    ]

    # Print each valid sequence joined by spaces.
    # We use print(*item) inside a list comprehension to trigger the side effect of printing.
    # The outer list comprehension is used solely to iterate through the results.
    [print(*(seq)) for seq in results]

if __name__ == "__main__":
    solve()