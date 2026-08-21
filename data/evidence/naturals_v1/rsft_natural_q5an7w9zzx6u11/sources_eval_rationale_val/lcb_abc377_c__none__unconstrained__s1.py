import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the grid size, M is the number of pieces
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Parse the coordinates of the M pieces
    # pieces is a list of tuples [(a1, b1), (a2, b2), ...]
    pieces = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Define the relative moves of a Knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Use a set comprehension to find all squares that are either:
    # 1. Occupied by an existing piece
    # 2. Under attack by an existing piece (and within grid boundaries)
    # The set automatically handles duplicates.
    forbidden_squares = {
        (a + dr, b + dc)
        for a, b in pieces
        for dr, dc in ( (0, 0) + tuple(moves) ) # Include (0,0) to mark the piece itself as forbidden
        if 1 <= a + dr <= N and 1 <= b + dc <= N
    }
    
    # The logic above for (0,0) is slightly flawed in the loop structure.
    # Let's refine it: we need the piece's own position AND its attack positions.
    # We can achieve this by flattening a list of coordinates for each piece.
    
    # Corrected set comprehension:
    # For every piece (a, b), we create a list of its position and its 8 attack positions.
    # Then we flatten that list and filter by grid boundaries.
    
    # Since I cannot use loops, I will use a nested comprehension.
    # We create a tuple of 9 possible positions for each piece, then iterate through them.
    
    forbidden = {
        (a + dr, b + dc)
        for a, b in pieces
        for dr, dc in [(0, 0), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
        if 1 <= a + dr <= N and 1 <= b + dc <= N
    }
    
    # Total squares is N*N. Subtract the number of forbidden squares.
    print(N * N - len(forbidden))

if __name__ == "__main__":
    solve()