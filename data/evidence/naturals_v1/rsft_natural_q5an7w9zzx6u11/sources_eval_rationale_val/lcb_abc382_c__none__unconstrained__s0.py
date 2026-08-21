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
    
    # Create a list of (index, gourmet_level) for the people
    people = list(enumerate(A, 1))
    
    # For each sushi, find the first person i such that B_j >= A_i
    # next(generator, default) is used to find the first match without a loop
    # The generator expression iterates through the people list
    results = [
        next((person_idx for person_idx, level in people if sushi_val >= level), -1)
        for sushi_val in B
    ]
    
    # Print each result on a new line
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()