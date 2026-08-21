import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of people, M: number of sushi
    N = int(input_data[0])
    M = int(input_data[1])
    
    # A: Gourmet levels of people
    # B: Deliciousness of sushi
    A = input_data[2:2+N]
    B = input_data[2+N:]

    # We need to find the first index i such that B_j >= A_i.
    # We use a generator expression inside next() to find the first match.
    # enumerate(A, 1) provides (index, value) pairs starting from 1.
    # next(..., -1) returns the index if found, otherwise -1.
    
    # Pre-convert A to integers to avoid repeated casting inside the generator
    A_ints = list(map(int, A))
    
    def find_eater(b_val):
        b_int = int(b_val)
        return next((i for i, a in enumerate(A_ints, 1) if b_int >= a), -1)

    # Use map to apply find_eater to every sushi in B
    results = map(find_eater, B)
    
    # Use sys.stdout.write and join to output results without a for loop
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()