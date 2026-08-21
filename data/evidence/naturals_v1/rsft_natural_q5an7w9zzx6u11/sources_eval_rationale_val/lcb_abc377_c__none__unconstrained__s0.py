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
        (int(input_data[2 + 2*i]), int(input_data[3 + 2*i])) 
        for i in range(M)
    ]
    
    # Define the relative moves of a Knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # For every piece, calculate all squares it can capture.
    # We use a set comprehension to automatically handle duplicates.
    # We filter for squares that are within the grid boundaries [1, N].
    # We also include the squares where pieces are already placed, 
    # as the problem states we must place the piece on an "empty" square.
    
    # 1. Squares occupied by existing pieces
    occupied = set(pieces)
    
    # 2. Squares threatened by existing pieces
    # We use a nested comprehension: for each piece, for each move, calculate the target square.
    threatened = {
        (a + da, b + db)
        for (a, b) in pieces
        for (da, db) in moves
        if 1 <= a + da <= N and 1 <= b + db <= N
    }
    
    # The total number of unavailable squares is the union of occupied and threatened.
    # The number of available squares is Total - size of (occupied UNION threatened).
    # Since occupied is a subset of the total grid and threatened is also a subset,
    # we just need the size of the union.
    
    # Using the | operator for set union
    unavailable_count = len(occupied | threatened)
    
    # Total squares N^2 minus unavailable squares
    print(N * N - unavailable_count)

if __name__ == "__main__":
    solve()