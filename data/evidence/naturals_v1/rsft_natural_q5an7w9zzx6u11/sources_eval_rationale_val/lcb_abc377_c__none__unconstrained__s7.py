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
    # Using a list comprehension to create pairs of (a, b)
    pieces = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(2, len(input_data), 2)
    ]
    
    # Define the 8 relative moves of a knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # For every piece, find all squares it can attack.
    # We use a set comprehension to store all attacked squares.
    # A square (r, c) is attacked if it's within grid boundaries.
    attacked_squares = {
        (r + dr, c + dc)
        for r, c in pieces
        for dr, dc in moves
        if 1 <= r + dr <= N and 1 <= c + dc <= N
    }
    
    # The piece cannot be placed on a square already occupied by an existing piece.
    # We add the existing pieces to the set of unavailable squares.
    # Since we need to subtract the total unavailable from N^2, 
    # we can just union the sets.
    unavailable_squares = attacked_squares.union(set(pieces))
    
    # Total squares is N*N. 
    # Result is N*N minus the number of unique unavailable squares.
    print(N * N - len(unavailable_squares))

if __name__ == "__main__":
    solve()