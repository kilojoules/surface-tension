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

    # We need to find the first person i such that B_j >= A_i.
    # Since we cannot use for/while loops, we use a generator expression
    # inside next() to find the first matching index.
    # enumerate(A, 1) provides (index, value) pairs starting from 1.
    
    def find_eater(sushi_val):
        # next(generator, default) returns the first item from the generator
        # or the default value if the generator is exhausted.
        return next(
            (person_idx for person_idx, gourmet_lvl in enumerate(A, 1) 
             if sushi_val >= gourmet_lvl), 
            -1
        )

    # Use map to apply find_eater to every piece of sushi in B.
    results = map(find_eater, B)
    
    # Use sys.stdout.write and join to output results without a loop.
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()