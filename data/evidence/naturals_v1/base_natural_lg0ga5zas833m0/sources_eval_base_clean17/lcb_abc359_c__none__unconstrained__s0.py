import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy = map(int, line[0:2])
        tx, ty = map(int, line[2:4])
    except EOFError:
        return

    # The grid is covered by 2x1 tiles.
    # Rule: If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed row j:
    # If j is even, tiles are [0,1], [2,3], [4,5]... (A_{i,j} and A_{i+1,j} are same if i is even)
    # If j is odd, tiles are [1,2], [3,4], [5,6]... (A_{i,j} and A_{i+1,j} are same if i is odd)
    
    # Let's define a coordinate system for the tiles.
    # A square (i, j) belongs to a tile.
    # If i + j is even, (i, j) and (i+1, j) are the same tile.
    # Let's map each square (i, j) to a "tile coordinate" (U, V).
    # For a fixed j, the squares are grouped in pairs.
    # If j is even: (0,j)&(1,j), (2,j)&(3,j)... -> U = i // 2
    # If j is odd: (-1,j)&(0,j), (1,j)&(2,j)... -> U = (i - 1) // 2
    # This is equivalent to: U = (i + (j % 2)) // 2
    # Now for the vertical direction:
    # Each square (i, j) is in exactly one tile. The rule only defines horizontal merges.
    # So vertical movement always moves to a different tile unless we stay in the same square.
    # However, the problem says "Each time he enters a tile, he pays a toll of 1".
    # Starting tile is free.
    
    # Let's simplify:
    # The distance in the "tile grid" is the Manhattan distance.
    # Let X(i, j) = (i + (j % 2)) // 2
    # Let Y(i, j) = j
    # Moving from (sx, sy) to (tx, ty):
    # Start tile: (X(sx, sy), Y(sx, sy))
    # End tile: (X(tx, ty), Y(tx, ty))
    
    # But wait, the movement is a bit more flexible.
    # Moving horizontally within a tile costs 0.
    # Moving vertically always enters a new tile.
    # Let's re-evaluate the distance.
    # To go from (sx, sy) to (tx, ty):
    # Vertical distance: |sy - ty|
    # Horizontal distance: 
    # At row j, the tile index is (i + (j % 2)) // 2.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # cost = |sy - ty| + max(0, distance_between_tile_columns)
    # Actually, the simplest way to think about this is:
    # Each step in Y costs 1.
    # Each step in X costs 1, but we can move "for free" if we are in the same tile.
    # The tiles are arranged in a brick-like pattern.
    # The distance between (sx, sy) and (tx, ty) in this specific grid is:
    # dist = |sy - ty| + max(0, |X(sx, sy) - X(tx, ty)| - (1 if sy and ty have different parity and we can optimize))
    # Actually, the formula for this specific grid distance is:
    # cost = abs(sy - ty) + max(0, abs(X(sx, sy) - X(tx, ty)) - (1 if abs(sy - ty) > 0 else 0))
    # Let's check Sample 1: (5, 0) to (2, 5)
    # X(5, 0) = (5 + 0)//2 = 2
    # X(2, 5) = (2 + 1)//2 = 1
    # |0 - 5| = 5. |2 - 1| = 1.
    # cost = 5 + max(0, 1 - 1) = 5. Correct.
    
    # Sample 2: (3, 1) to (4, 1)
    # X(3, 1) = (3 + 1)//2 = 2
    # X(4, 1) = (4 + 1)//2 = 2
    # |1 - 1| = 0. |2 - 2| = 0.
    # cost = 0 + max(0, 0 - 0) = 0. Correct.
    
    # Let's refine the logic:
    # Let u1 = (sx + (sy % 2)) // 2
    # Let v1 = sy
    # Let u2 = (tx + (ty % 2)) // 2
    # Let v2 = ty
    
    # The distance is |v1 - v2| + max(0, |u1 - u2| - (1 if v1 != v2 else 0))
    # Wait, if v1 != v2, we can move diagonally in the tile grid.
    # If we move from (u1, v1) to (u1, v1+1), we pay 1.
    # From (u1, v1+1), the horizontal boundary is shifted.
    # The distance is actually:
    # cost = abs(v1 - v2) + max(0, abs(u1 - u2) - (1 if abs(v1 - v2) > 0 else 0))
    # But we must be careful. If we move vertically, we can potentially change our U coordinate
    # relative to the tiles.
    # Let's use the property:
    # cost = max(abs(v1 - v2), abs(u1 - u2) + (0 if v1 == v2 else -1))
    # No, the simplest correct formula for this grid is:
    # cost = abs(v1 - v2) + max(0, abs(u1 - u2) - (1 if abs(v1 - v2) > 0 else 0))
    # Let's re-verify Sample 1: 5 + max(0, 1 - 1) = 5.
    # Sample 2: 0 + max(0, 0 - 0) = 0.
    # Another case: (0,0) to (1,0). X(0,0)=0, X(1,0)=0. cost = 0 + 0 = 0.
    # (0,0) to (2,0). X(0,0)=0, X(2,0)=1. cost = 0 + max(0, 1-0) = 1.
    # (0,0) to (0,1). X(0,0)=0, X(0,1)=0. cost = 1 + 0 = 1.
    # (0,0) to (1,1). X(0,0)=0, X(1,1)=1. cost = 1 + max(0, 1-1) = 1.
    # (0,0) to (2,1). X(0,0)=0, X(2,1)=1. cost = 1 + max(0, 1-1) = 1.
    # (0,0) to (3,1). X(0,0)=0, X(3,1)=2. cost = 1 + max(0, 2-1) = 2.
    
    u1 = (sx + (sy % 2)) // 2
    v1 = sy
    u2 = (tx + (ty % 2)) // 2
    v2 = ty
    
    ans = abs(v1 - v2) + max(0, abs(u1 - u2) - (1 if abs(v1 - v2) > 0 else 0))
    print(ans)

if __name__ == "__main__":
    solve()