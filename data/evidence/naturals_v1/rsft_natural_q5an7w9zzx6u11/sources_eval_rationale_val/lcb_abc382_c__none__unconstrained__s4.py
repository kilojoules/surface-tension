import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N and M
    N = int(input_data[0])
    M = int(input_data[1])
    
    # A is the list of gourmet levels
    # B is the list of sushi deliciousness
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:]))
    
    # Create a list of (index, gourmet_level) to keep track of person numbers
    people = list(enumerate(A, 1))
    
    # For each sushi B_j, we want the first person i where B_j >= A_i.
    # We use a generator expression inside next() to find the first match.
    # next(generator, default) returns the first item or the default if empty.
    
    def find_eater(sushi_val):
        return next(
            (person_idx for person_idx, level in people if sushi_val >= level), 
            -1
        )
    
    # Use map to apply find_eater to every sushi in B
    results = map(find_eater, B)
    
    # Print all results separated by newlines
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()