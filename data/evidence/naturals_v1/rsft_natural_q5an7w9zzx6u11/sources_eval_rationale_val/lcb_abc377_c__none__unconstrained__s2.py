import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the grid size, M is the number of pieces
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
    attacked_or_occupied = {
        (r + dr, c + dc)
        for r, c in pieces
        for dr, dc in moves
        if 1 <= r + dr <= N and 1 <= c + dc <= N
    } | set(pieces)
    
    # The number of safe squares is Total Squares - Number of attacked/occupied squares
    # Total squares = N * N
    # The result can be very large, so we use Python's arbitrary precision integers
    print(N * N - len(attacked_or_occupied))

if __name__ == "__main__":
    solve()