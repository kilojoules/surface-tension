import sys

def solve():
    # Read all input at once and split into a list of strings
    input_data = sys.stdin.read().split()
    
    # N is the first element
    # A is the slice from index 1 to N+1
    # B is the slice from index N+1 to 2N+1
    # Using map(int, ...) to convert strings to integers without explicit loops
    n = int(input_data[0])
    a = list(map(int, input_data[1 : n + 1]))
    b = list(map(int, input_data[n + 1 : 2 * n + 1]))
    
    # The maximum value of A_i + B_j is simply the sum of the maximums of A and B
    # max() is a built-in that operates on the iterable without a Python-level loop
    print(max(a) + max(b))

if __name__ == "__main__":
    solve()