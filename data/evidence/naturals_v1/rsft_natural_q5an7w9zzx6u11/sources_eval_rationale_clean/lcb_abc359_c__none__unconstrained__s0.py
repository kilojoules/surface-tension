import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)
    
    # The tiling pattern:
    # When i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are horizontal 2x1 blocks.
    # In row j, if j is even, tiles are [0,1], [2,3]... (i=0, 2, 4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2]... (i=1, 3, 5...)
    # Wait, the rule is: if i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # Let's analyze:
    # For a fixed j:
    # If j is even: i=0, 2, 4... satisfy i+j is even. So (0,j)&(1,j), (2,j)&(3,j) are tiles.
    # If j is odd: i=1, 3, 5... satisfy i+j is even. So (1,j)&(2,j), (3,j)&(4,j) are tiles.
    
    # Let's define the tile ID for square (i, j):
    # If (i+j) is even, the tile is identified by (i // 2 if j%2==0 else (i-1)//2, j)
    # Actually, a simpler way:
    # A square (i, j) belongs to a horizontal tile if (i+j) is even (it's the left half)
    # or (i+j-1) is even (it's the right half).
    # The tile index in row j is (i if (i+j)%2 != 0 else i-1) // 2.
    # Let's use the property: 
    # In row j, the boundary between tiles occurs at x = k where (k+j) is odd.
    # The cost to move from (sx, sy) to (tx, ty):
    # Moving vertically always crosses a tile boundary unless you stay in the same tile.
    # But tiles are only 2x1 (horizontal). Moving vertically always enters a new tile.
    # Moving horizontally might stay in the same tile.
    
    # Let's normalize the coordinates.
    # In row j, the tiles are blocks of 2. 
    # If j is even, blocks are [0,1], [2,3]...
    # If j is odd, blocks are [-1,0], [1,2]...
    # We can transform x based on j: x' = x if j is even else x-1.
    # Then tiles are always [2k, 2k+1].
    # The tile index is x' // 2.
    
    # Let's use a helper to get the tile coordinate (tx, ty)
    # For a point (x, y), the tile is:
    # ty = y
    # tx = (x if y % 2 == 0 else x - 1) // 2
    
    # The distance between (sx', sy) and (tx', ty) in this transformed grid:
    # Moving from (sx', sy) to (tx', ty):
    # Each step in y costs 1.
    # Each step in x' costs 1, BUT we can move diagonally in the original grid.
    # Actually, the problem is simpler:
    # The cost is the Manhattan distance in the "tile graph".
    # In the tile graph, two tiles are adjacent if they share an edge.
    # The distance is max(|sx' - tx'|, |sy - ty|) if we can move diagonally? 
    # No, the rules say move n units Up, Down, Left, Right.
    # This is Manhattan distance in the tile graph.
    # However, moving 1 unit vertically might land you in a tile that is 
    # "closer" to the target x' than the current tile.
    
    # Correct logic for this specific tiling:
    # The distance is (|sy - ty| + |sx' - tx'|) / 2 rounded up? 
    # No. Let's use the formula:
    # cost = max(|sy - ty|, (|sy - ty| + abs(sx' - tx')) // 2) 
    # Wait, the simplest derivation for this grid is:
    # cost = abs(sy - ty) + max(0, (abs(sx' - tx') - abs(sy - ty) + 1) // 2)
    # Which simplifies to: cost = max(abs(sy - ty), (abs(sy - ty) + abs(sx' - tx')) // 2)
    # Let's re-verify. If sy=ty, cost is abs(sx'-tx').
    # If sx'=tx', cost is abs(sy-ty).
    # If we move 1 unit vertically, we can potentially change our x' by 1.
    # So we can cover 1 unit of x' distance for every 1 unit of y distance.
    # The total cost is abs(sy - ty) + max(0, abs(sx' - tx') - abs(sy - ty))
    # But we can move 2 units of x' for 1 unit of cost (since tiles are 2x1).
    # Actually, the cost is:
    # dx = abs(sx' - tx')
    # dy = abs(sy - ty)
    # result = dy + max(0, (dx - dy + 1) // 2)
    # This is equivalent to: (dx + dy + 1) // 2 if dx > dy else dy.
    # Which is max(dy, (dx + dy + 1) // 2).
    
    # Let's calculate sx', tx'
    # sx_prime = (sx if sy % 2 == 0 else sx - 1) // 2
    # tx_prime = (tx if ty % 2 == 0 else tx - 1) // 2
    
    # Using a list comprehension to avoid assignments and a final print
    # to satisfy the "no loop/recursion" and "single expression" feel.
    
    res = (lambda s_x, s_y, t_x, t_y: (
        lambda sxp, txp: (
            lambda dx, dy: max(dy, (dx + dy + 1) // 2)
        )(abs(sxp - txp), abs(s_y - t_y))
    )((s_x if s_y % 2 == 0 else s_x - 1) // 2, (t_x if t_y % 2 == 0 else t_x - 1) // 2))(sx, sy, tx, ty)
    
    print(res)

if __name__ == "__main__":
    solve()