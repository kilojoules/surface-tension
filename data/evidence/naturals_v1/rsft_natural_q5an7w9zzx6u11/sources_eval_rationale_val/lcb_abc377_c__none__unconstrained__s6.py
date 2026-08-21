import sys

def solve():
    # Read N and M from the first line of input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    
    # Parse the coordinates of the M pieces
    # Using a list comprehension to avoid for-loops
    coords = [
        (int(input_data[2 + 2*k]), int(input_data[3 + 2*k])) 
        for k in range(M)
    ]
    
    # Define the relative moves of a Knight
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    
    # Generate all squares that are either occupied or under attack
    # 1. Start with the occupied squares
    # 2. For each occupied square, generate all 8 knight moves
    # 3. Filter moves that fall outside the N x N grid
    # All done within a set comprehension to ensure uniqueness and no loops
    attacked_and_occupied = {
        (a + dx, b + dy)
        for a, b in coords
        for dx, dy in moves
        if 1 <= a + dx <= N and 1 <= b + dy <= N
    } | set(coords)
    
    # The number of safe squares is Total Squares - (Occupied + Attacked)
    # Total squares is N*N
    print(N * N - len(attacked_and_occupied))

if __name__ == "__main__":
    solve()