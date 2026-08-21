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
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:]))

    # We need to find the first person i (1-indexed) such that B_j >= A_i.
    # We use a generator expression inside next() to find the first match.
    # next(generator, default) returns the first item or the default if the generator is exhausted.
    
    # To avoid a loop inside the helper, we use enumerate to get indices and 
    # a generator expression to filter for the first person who can eat the sushi.
    def find_eater(sushi_val):
        return next(
            (idx + 1 for idx, gourmet_level in enumerate(A) if sushi_val >= gourmet_level), 
            -1
        )

    # Use map to apply the find_eater function to every piece of sushi in B.
    results = map(find_eater, B)

    # Use sys.stdout.write and join to output results without a for loop.
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()