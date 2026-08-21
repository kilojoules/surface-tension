import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Parse pieces into a list of (r, c) tuples
    # Using a list comprehension to avoid for-loops
    pieces = [
        (int(input_data[2 + 2*k]), int(input_data[3 + 2*k])) 
        for k in range(M)
    ]
    
    # Define the 8 relative moves of a knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Generate all squares that are either occupied or under attack
    # 1. Start with the occupied squares
    # 2. Add all valid squares reachable by a knight move from each piece
    # We use a set comprehension to automatically handle duplicates
    forbidden = {
        (r + dr, c + dc)
        for r, c in pieces
        for dr, dc in moves
        if 1 <= r + dr <= N and 1 <= c + dc <= N
    }
    
    # Add the squares where pieces are already placed to the forbidden set
    # Since we can't use .update() in a loop, we use the union operator |
    forbidden = forbidden | set(pieces)
    
    # Total squares is N*N
    # Result is Total - number of forbidden squares
    print(N * N - len(forbidden))

if __name__ == "__main__":
    solve()