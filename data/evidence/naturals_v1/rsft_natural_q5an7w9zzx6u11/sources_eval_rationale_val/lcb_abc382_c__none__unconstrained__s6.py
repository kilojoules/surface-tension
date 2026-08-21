import sys

def solve():
    # Read all input at once and split into a list of integers
    data = list(map(int, sys.stdin.read().split()))
    
    # N: number of people, M: number of sushi
    n, m = data[0], data[1]
    
    # A: gourmet levels of people
    # B: deliciousness of sushi
    a = data[2:2+n]
    b = data[2+n:]
    
    # Create a list of (index, gourmet_level) for the people
    # People are 1-indexed
    people = list(enumerate(a, 1))
    
    # For each sushi B_j, find the first person i where B_j >= A_i
    # next(generator, default) is used to find the first match or return -1
    results = [
        next((person_id for person_id, level in people if sushi_val >= level), -1)
        for sushi_val in b
    ]
    
    # Print each result on a new line
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()