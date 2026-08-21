import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # N: number of people, M: number of sushi
    N = int(input_data[0])
    M = int(input_data[1])
    
    # A: gourmet levels of people
    # B: deliciousness of sushi
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:]))

    # We need to find the first person i (1-indexed) such that B_j >= A_i.
    # We can use a generator expression inside next() to find the first index.
    # next(generator, default) returns the first item or the default if the generator is exhausted.
    
    # To avoid a loop over M, we use map() to apply the logic to every sushi in B.
    # To avoid a loop over N, we use a generator expression inside next().
    
    # We create a indexed version of A to track person numbers: (gourmet_level, person_number)
    people = list(enumerate(A, 1))
    
    def find_eater(sushi_val):
        # Find the first person whose gourmet level is <= sushi deliciousness
        # next() effectively replaces the 'for' loop and 'break'
        return next((person_num for level, person_num in people if sushi_val >= level), -1)

    # Use map to process all sushi and join the results with newlines
    results = map(find_eater, B)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()