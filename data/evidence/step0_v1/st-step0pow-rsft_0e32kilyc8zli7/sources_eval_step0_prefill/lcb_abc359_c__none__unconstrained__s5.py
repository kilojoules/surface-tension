import sys

def solve():
    # Read input from standard input
    try:
        line = sys.stdin.read().split()
        if not line:
            return
        sx, sy, tx, ty = map(int, line)
    except EOFError:
        return

    # The rule for tiling is:
    # If i + j is even, A_{i,j} and A_{i+1,j} are in the same tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), (4,5)... if j is even
    # and pairs (1,2), (3,4), (5,6)... if j is odd.
    #
    # Let's analyze the cost to move.
    # Moving horizontally within a tile costs 0.
    # Moving vertically always enters a new tile unless the destination is the same tile.
    # However, the problem says "Each time he enters a tile, he pays a toll of 1."
    # This implies the starting tile is already "entered" or we only pay for transitions.
    # Re-reading: "Each time he enters a tile, he pays a toll of 1."
    # Usually, this means the number of tiles visited minus 1.
    # Let's check Sample 2: (3,1) to (4,1).
    # i=3, j=1. i+j = 4 (even). A_{3,1} and A_{4,1} are in the same tile.
    # Moving from (3.5, 1.5) to (4.5, 1.5) stays in the same tile. Cost 0. Correct.
    
    # Let's define a coordinate transformation to make the grid uniform.
    # In the original grid, tiles are 2x1.
    # For a fixed j, the boundaries between tiles are at x = k where k is even if j is even, and k is odd if j is odd.
    # This is equivalent to saying the boundary is at x = k where k % 2 == j % 2.
    # Let's transform (x, y) -> (X, Y) such that the tiles are standard 1x1 squares.
    # If we shift x by (j % 2), the boundaries become at even integers.
    # Then we can divide x by 2.
    # Let X = (x + (y % 2)) // 2
    # Let Y = y
    # Now, a move in Y always changes the tile. A move in X changes the tile if X changes.
    # The distance is |X1 - X2| + |Y1 - Y2|.
    # But wait, the cost is the number of tiles entered.
    # If we move from (X1, Y1) to (X2, Y2), the minimum number of tiles entered is
    # the Manhattan distance in the transformed grid, but we must be careful.
    # In the transformed grid, moving from (X, Y) to (X, Y+1) costs 1.
    # Moving from (X, Y) to (X+1, Y) costs 1.
    # However, we can move diagonally in the original grid by combining moves.
    # Actually, the most efficient way to move between (X1, Y1) and (X2, Y2) 
    # in a grid of 1x1 tiles is the Manhattan distance: |X1 - X2| + |Y1 - Y2|.
    # But we can "shortcut" if we can move to a tile that is adjacent both horizontally and vertically.
    # In this specific tiling, the tiles are 2x1.
    # Let's re-evaluate.
    # The cost to move from (sx, sy) to (tx, ty) is:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # If we move only vertically, cost is dy.
    # If we move only horizontally, cost is dx // 2 (roughly).
    # The optimal strategy is to move diagonally as much as possible.
    # One "diagonal" step (changing both X and Y) costs 1 toll if we enter a tile that 
    # covers the transition.
    # Specifically, if we are at (X, Y), moving to (X+1, Y+1) can be done by:
    # (X, Y) -> (X, Y+1) [cost 1] -> (X+1, Y+1) [cost 1] = 2
    # OR (X, Y) -> (X+1, Y) [cost 1] -> (X+1, Y+1) [cost 1] = 2
    # Wait, the sample 1: (5,0) to (2,5). 
    # sx=5, sy=0, tx=2, ty=5.
    # X1 = (5 + 0)//2 = 2. Y1 = 0.
    # X2 = (2 + 5%2)//2 = (2+1)//2 = 1. Y2 = 5.
    # Manhattan distance = |2-1| + |5-0| = 1 + 5 = 6.
    # Sample 1 output is 5. 
    # This means we can move from (X, Y) to (X+1, Y+1) with cost 1.
    # This happens if the tile at (X+1, Y+1) is adjacent to the tile at (X, Y).
    # In our transformed grid, the tiles are 1x1. 
    # The original tiles are 2x1. 
    # A tile is {(x,y) | i <= x <= i+2, j <= y <= j+1} where i+j is even.
    # This tile covers (X, Y) in the transformed coordinates.
    # The tiles are effectively 1x1 in (X, Y) space.
    # The cost to move from (X1, Y1) to (X2, Y2) is max(|X1 - X2|, |Y1 - Y2|).
    # Let's check Sample 1: X1=2, Y1=0, X2=1, Y2=5. max(|2-1|, |5-0|) = 5. Correct.
    # Sample 2: (3,1) to (4,1). X1=(3+1)//2=2, Y1=1. X2=(4+1)//2=2, Y2=1. max(0,0)=0. Correct.
    
    # Final logic:
    # X = (x + (y % 2)) // 2
    # Y = y
    # Result = max(abs(X1 - X2), abs(Y1 - Y2))
    
    x1 = (sx + (sy % 2)) // 2
    y1 = sy
    x2 = (tx + (ty % 2)) // 2
    y2 = ty
    
    print(max(abs(x1 - x2), abs(y1 - y2)))

if __name__ == "__main__":
    solve()