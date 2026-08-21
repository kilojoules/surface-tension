import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiles are 2x1 blocks. 
    # When i+j is even, A_{i,j} and A_{i+1,j} are the same tile.
    # This means for a fixed j, tiles are paired as (0,1), (2,3), etc. if j is even,
    # and (-1,0), (1,2), etc. if j is odd.
    # Effectively, a tile's identity is defined by (floor((i + (j % 2)) / 2), j).
    
    # Let's transform the coordinates to a system where we can calculate 
    # the distance in terms of tile transitions.
    # In the given layout, moving vertically always crosses a tile boundary.
    # Moving horizontally might not if you stay within the 2x1 block.
    
    # The cost to move from (sx, sy) to (tx, ty) is:
    # 1. The vertical distance |sy - ty|.
    # 2. The horizontal distance, but since tiles are 2 units wide, 
    #    it's roughly |sx - tx| / 2.
    # Specifically, the cost is max(|sy - ty|, (|sx - tx| + 1) // 2) 
    # if we can optimize, but the actual rule is:
    # Cost = |sy - ty| + max(0, (|sx - tx| + 1) // 2 - |sy - ty|) 
    # is not quite right because vertical moves can 'cover' horizontal distance.
    
    # The correct observation for this specific tiling:
    # The distance is max(|sy - ty|, (|sx - tx| + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Wait, the simplest closed form for this specific grid problem is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # if we consider the parity of the starting tile.
    
    # Let's refine:
    # A move in Y always costs 1.
    # A move in X costs 1 per 2 units.
    # The minimum cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # ONLY if the parity allows.
    
    # The actual minimum toll is:
    # Let dx = abs(sx - tx)
    # Let dy = abs(sy - ty)
    # The cost is max(dy, (dx + 1) // 2) if we can move diagonally.
    # But we can only move in 4 directions.
    # However, a sequence of (Right, Up, Right, Up) is equivalent to 
    # moving diagonally in terms of cost.
    
    # The correct formula for this problem is:
    # result = max(abs(sy - ty), (abs(sx - tx) + 1) // 2)
    # But we must check if the start and end points are in the same tile.
    # If sx, sy and tx, ty are in the same tile, cost is 0.
    
    # Two points (sx, sy) and (tx, ty) are in the same tile if:
    # sy == ty AND (sx + sy) // 2 == (tx + ty) // 2
    
    # For the general case:
    # The cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2)
    # Let's test Sample 1: 5 0, 2 5 -> dx=3, dy=5. max(5, (3+1)//2) = 5. Correct.
    # Sample 2: 3 1, 4 1 -> dx=1, dy=0. max(0, (1+1)//2) = 1? 
    # Wait, Sample 2 output is 0. 
    # In Sample 2: sx=3, sy=1. i+j = 3+1 = 4 (even). 
    # A_{3,1} and A_{4,1} are the same tile. So cost is 0.
    
    # The condition for being in the same tile:
    # sy == ty and ((sx + sy) // 2 == (tx + ty) // 2) if (sx+sy)%2 == 0
    # Actually, the rule is: if i+j is even, (i,j) and (i+1,j) are one tile.
    # This means tile ID is ( (i + (j%2)) // 2, j )
    
    # Let's use the tile ID logic:
    # start_tile = ((sx + (sy % 2)) // 2, sy)
    # end_tile = ((tx + (ty % 2)) // 2, ty)
    # dx_tile = abs(start_tile[0] - end_tile[0])
    # dy_tile = abs(start_tile[1] - end_tile[1])
    # The cost is max(dy_tile, (dx_tile + 1) // 2) is still not quite right.
    # The cost is actually just the distance in a transformed coordinate system.
    # The distance is max(dy_tile, (dx_tile + 1) // 2) is for a different problem.
    # For this problem, the cost is simply:
    # max(dy_tile, (dx_tile + 1) // 2) is for 8-connectivity.
    # With 4-connectivity, the cost is dy_tile + dx_tile if we can't move diagonally.
    # But we can move n units. This means we can change X and Yに 
    # independently.
    # The cost is dy_tile + (dx_tile + 1) // 2 is also wrong.
    
    # Let's re-evaluate:
    # To change Y by 1, cost is 1.
    # To change X by 2, cost is 1 (by moving Y then X then Y back).
    # The minimum cost is max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # is only if you can move diagonally.
    # But you can move: (0,0) -> (0,1) [cost 1] -> (1,1) [cost 0] -> (1,0) [cost 1]
    # That's 2 units of X for 2 cost.
    # Actually, the simplest answer is:
    # cost = max(abs(sy - ty), (abs(sx - tx) + 1) // 2) 
    # Wait, Sample 2: sx=3, sy=1, tx=4, ty=1.
    # start_tile = ((3 + 1)//2, 1) = (2, 1)
    # end_tile = ((4 + 1)//2, 1) = (2, 1)
    # distance = max(0, (0+1)//2) = 0. Correct.
    
    # Let's check Sample 1: 5 0, 2 5
    # start_tile = ((5 + 0)//2, 0) = (2, 0)
    # end_tile = ((2 + 5)//2, 5) = (3, 5)
    # dx_tile = abs(2 - 3) = 1
    # dy_tile = abs(0 - 5) = 5
    # max(5, (1+1)//2) = 5. Correct.
    
    # Final formula:
    # s_tile_x = (sx + (sy % 2)) // 2
    # t_tile_x = (tx + (ty % 2)) // 2
    # print(max(abs(sy - ty), (abs(s_tile_x - t_tile_x) + 1) // 2))
    # Wait, the (dx+1)//2 is for when you can move diagonally.
    # With 4-direction moves, you can simulate a diagonal move:
    # (0,0) -> (0,1) [cost 1] -> (1,1) [cost 0]
    # This moves you 1 unit X and 1 unit Y for cost 1.
    # So the cost is indeed max(abs(sy - ty), (abs(s_tile_x - t_tile_x) + 1) // 2)
    # No, that's not right. If dx_tile is very large, you need more cost.
    # The cost to move 1 unit in tile_x is 1 (by moving Y then X).
    # So the cost is max(dy_tile, dx_tile) if you can move diagonally.
    # But you can't. However, you can move (0,0) -> (0,1) -> (1,1) 
    # which is 1 unit Y and 1 unit tile_X for cost 1.
    # Thus, the cost is max(dy_tile, dx_tile).
    # Let's check Sample 1: dy=5, dx_tile=1. max(5, 1) = 5.
    # Sample 2: dy=0, dx_tile=0. max(0, 0) = 0.
    # Sample 3: 
    # sx=2552608206527595, sy=5411232866732612
    # tx=771856005518028, ty=7206210729152763
    # s_tile_x = (2552608206527595 + 0) // 2 = 1276304103263797
    # t_tile_x = (771856005518028 + 1) // 2 = 385928002759014
    # dx_tile = 1276304103263797 - 385928002759014 = 890376100504783
    # dy_tile = abs(5411232866732612 - 7206210729152763) = 1794977862420151
    # max(1794977862420151, 890376100504783) = 1794977862420151. Correct.

    s_tile_x = (sx + (sy % 2)) // 2
    t_tile_x = (tx + (ty % 2)) // 2
    print(max(abs(sy - ty), abs(s_tile_x - t_tile_x)))

if __name__ == "__main__":
    solve()