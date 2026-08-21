import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Parse coordinates into a list of tuples
    # Using a list comprehension to avoid for-loops
    coords = [
        (int(input_data[2 + 2*i]), int(input_data[3 + 2*i])) 
        for i in range(M)
    ]
    
    # Define the 8 relative moves of a knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Generate all squares that are either occupied or under attack
    # 1. Squares occupied by existing pieces
    # 2. Squares reachable by a knight move from existing pieces
    # We use a set comprehension to automatically handle duplicates
    forbidden = {
        (r + dr, c + dc)
        for r, c in coords
        for dr, dc in moves
        if 1 <= r + dr <= N and 1 <= c + dc <= N
    }
    
    # Add the occupied squares to the forbidden set
    # Since we cannot use .update() in a loop, we merge sets
    forbidden |= set(coords)
    
    # Total squares is N*N
    # Result is total squares minus the number of forbidden squares
    print(N * N - len(forbidden))

if __name__ == "__main__":
    solve()