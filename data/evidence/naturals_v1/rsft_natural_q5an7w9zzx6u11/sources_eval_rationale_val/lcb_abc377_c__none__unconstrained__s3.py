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
    
    # Generate all squares threatened by the existing pieces
    # We use a nested comprehension: for every piece, for every move.
    # We filter for squares that stay within the N x N grid boundaries.
    threatened_squares = {
        (r + dr, c + dc)
        for r, c in pieces
        for dr, dc in moves
        if 1 <= r + dr <= N and 1 <= c + dc <= N
    }
    
    # The piece we place cannot be on a square already occupied by a piece.
    # The problem asks for empty squares that cannot be captured.
    # A square is "invalid" if it is occupied OR if it is threatened.
    # Note: A piece at (r, c) threatens (r+dr, c+dc). 
    # If we place our piece at (r+dr, c+dc), it is captured.
    
    # We need to find the number of squares that are:
    # 1. Not occupied by the M pieces.
    # 2. Not threatened by any of the M pieces.
    
    # Let S be the set of occupied squares.
    # Let T be the set of threatened squares.
    # Total invalid squares = |S union T|.
    
    # Since we need to subtract occupied squares, we add the pieces to the set.
    all_invalid = threatened_squares.union(set(pieces))
    
    # Total squares is N*N.
    # Result is N^2 - number of invalid squares.
    print(N * N - len(all_invalid))

if __name__ == "__main__":
    solve()