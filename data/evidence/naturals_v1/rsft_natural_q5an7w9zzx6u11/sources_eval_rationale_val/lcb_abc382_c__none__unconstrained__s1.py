import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    
    # A is the list of gourmet levels, B is the list of sushi deliciousness
    # We use a list of tuples (index, value) for A to keep track of person numbers
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:]))
    
    # Create a indexed version of A to avoid using range() in a loop
    # indexed_A = [(1, A[0]), (2, A[1]), ...]
    indexed_A = list(zip(range(1, N + 1), A))
    
    # For each sushi b in B, find the first person (idx, val) in indexed_A such that b >= val
    # next(generator, default) is used to find the first match or return -1
    results = map(
        lambda b: next((idx for idx, val in indexed_A if b >= val), -1),
        B
    )
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()