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
    R = list(map(int, input_data[2:]))

    # Generate ranges for each R_i: [1, 2, ..., R_i]
    # We use map to create a list of ranges
    ranges = map(lambda r: range(1, r + 1), R)

    # itertools.product generates the Cartesian product of the input iterables.
    # It generates them in lexicographical order if the input iterables are sorted.
    all_sequences = itertools.product(*ranges)

    # Use a list comprehension to filter sequences where the sum is a multiple of K.
    # This replaces the need for a for-loop and an if-statement.
    valid_sequences = [seq for seq in all_sequences if sum(seq) % K == 0]

    # Use join and map to format the output without explicit loops.
    # We create a string for each sequence and then join them with newlines.
    output = "\n".join(
        " ".join(map(str, seq)) for seq in valid_sequences
    )

    # Print the final result if there are any valid sequences.
    if output:
        sys.stdout.write(output + "\n")

if __name__ == "__main__":
    solve()