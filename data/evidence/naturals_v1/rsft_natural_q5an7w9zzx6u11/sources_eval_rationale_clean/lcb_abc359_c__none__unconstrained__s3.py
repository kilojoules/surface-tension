The constraint to avoid explicit loops and recursion forces a shift toward functional programming paradigms, emphasizing the use of expressions over statements. This encourages the use of high-level abstractions and built-in mathematical functions, which can often be more concise and efficient for coordinate-based calculations. To solve this problem, I will use `abs()` for distance calculations and `max()` to determine the minimum toll based on the Manhattan distance and the specific tiling pattern.

```python
import sys

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy = map(int, input_data[0:2])
    tx, ty = map(int, input_data[2:4])

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means tiles are 2x1 horizontal blocks.
    # In row j, if j is even, tiles are [0,1], [2,3]... (i=0, 2, 4...)
    # In row j, if j is odd, tiles are [-1,0], [1,2]... (i=-1, 1, 3...)
    # Essentially, the "boundary" between tiles in row j is at x = k if k+j is odd.
    
    # Let's transform coordinates to a grid where each unit is a tile.
    # In row j, the tile index is (i + (j % 2)) // 2.
    # However, a simpler observation:
    # Moving vertically always crosses a tile boundary unless you stay in the same tile.
    # Moving horizontally might not cross a boundary.
    
    # The cost to move from (sx, sy) to (tx, ty):
    # Each vertical step of 1 unit always enters a new tile.
    # Total vertical cost = abs(sy - ty).
    # For horizontal movement, we only pay if we cross a boundary.
    # In a row j, boundaries are at x such that x+j is odd.
    # The number of boundaries between sx and tx in row j is:
    # If we are at (sx, sy) and (tx, ty), the most efficient way is to 
    # move vertically and horizontally.
    
    # Let dx = abs(sx - tx) and dy = abs(sy - ty).
    # The total cost is dy + (cost to move dx horizontally).
    # The horizontal cost depends on whether the start and end points 
    # are in the same tile relative to the row's offset.
    
    # A point (x, y) is in tile ( (x + (y % 2)) // 2, y )
    # Let's define a function to get the tile coordinate:
    # tile_x = (x + (y % 2)) // 2
    
    # The minimum cost is dy + max(0, (dist between tile_x of S and T) - (something))
    # Actually, the simplest derivation for this specific tiling is:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) + ( (sx+sy)%2 != (tx+ty)%2 )) // 2 - (some offset))
    # Let's refine:
    # The distance is dy + ceil( (dx - (1 if both are in the same tile type else 0)) / 2 )
    
    # Correct logic for this tiling:
    # The cost is dy + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0) + 1) // 2 - 1)
    # Wait, let's use the property:
    # Cost = abs(sy - ty) + max(0, (abs(sx - tx) + (1 if (sx+sy)%2 != (tx+ty)%2 else 0)) // 2)
    # Let's test Sample 1: 5 0 -> 2 5. dx=3, dy=5. 
    # (5+0)%2 = 1, (2+5)%2 = 1. Same parity.
    # Cost = 5 + (3 // 2) = 5 + 1 = 6? No, Sample 1 says 5.
    # If we move to a row where the horizontal distance is "cheaper".
    # In row j, the tile index is (x + (j%2)) // 2.
    # We want to minimize abs(sy - ty) + abs( (sx + (sy%2))//2 - (tx + (ty%2))//2 )
    # But we can change the row we perform the horizontal move in.
    # If we move to row j, the cost is abs(sy - j) + abs(ty - j) + abs((sx + (j%2))//2 - (tx + (j%2))//2).
    # This is minimized when j is between sy and ty.
    # The cost is dy + min( abs((sx + 0)//2 - (tx + 0)//2), abs((sx + 1)//2 - (tx + 1)//2) )
    
    # Let's check Sample 1: 5 0, 2 5. dy=5.
    # j=0: abs((5+0)//2 - (2+0)//2) = abs(2 - 1) = 1. Total = 6.
    # j=1: abs((5+1)//2 - (2+1)//2) = abs(3 - 1) = 2. Total = 7.
    # Wait, the sample says 5. How?
    # "Move left by 1 (toll 0), up by 1 (toll 1), left by 1 (toll 0)..."
    # This means he is utilizing the 2x1 tiles to move horizontally for free.
    # If he is in a tile that covers x and x+1, moving between them is free.
    # In row j, tiles are {(2k, j), (2k+1, j)} if j is even, and {(2k-1, j), (2k, j)} if j is odd.
    # This means in any row, he can move 1 unit horizontally for free.
    # To move dx units, he needs dx // 2 tiles, but he can use the "free" 
    # side of the tile he enters via a vertical move.
    
    # Correct logic:
    # Each vertical move enters a new tile.
    # To move dx horizontally, he needs to cross boundaries.
    # The number of boundaries is (dx + 1) // 2 if he starts/ends on boundaries.
    # The minimum cost is dy + max(0, (dx + (1 if (sx+sy)%2 == (tx+ty)%2 else 0)) // 2 - 1)
    # Let's re-evaluate Sample 1: dx=3, dy=5. (5+0)%2=1, (2+5)%2=1.
    # Cost = 5 + (3 + 1)//2 - 1 = 5 + 2 - 1 = 6. Still not 5.
    
    # Let's look at the sample path:
    # (5.5, 0.5) -> (4.5, 0.5) [Left 1, Toll 0] : Tile is {4,5} in row 0 (0+0 even, so {0,1},{2,3},{4,5})
    # (4.5, 0.5) -> (4.5, 1.5) [Up 1, Toll 1] : Enters tile {4,5} in row 1? No, row 1 tiles are {1,2},{3,4},{5,6}
    # In row 1, x=4 is in tile {3,4}.
    # Path: (5.5, 0.5) [Tile {4,5}, 0] -> (4.5, 0.5) [Tile {4,5}, 0] -> (4.5, 1.5) [Tile {3,4}, 1] 
    # -> (3.5, 1.5) [Tile {3,4}, 0] -> (3.5, 4.5) [Up 3, Toll 3] -> (2.5, 4.5) [Tile {2,3}, 0] 
    # -> (2.5, 5.5) [Up 1, Toll 1]. Total = 1+3+1 = 5.
    
    # Observation: He can move horizontally for free if he is within the 2x1 tile.
    # He can move vertically and enter a tile that allows him to move horizontally.
    # The cost is simply dy + (dx + 1) // 2, but he can save 1 if the start/end 
    # positions allow him to use the "free" horizontal move of the first/last tile.
    # Specifically, if (sx+sy)%2 == 0, he is at the left side of a tile.
    # If (sx+sy)%2 == 1, he is at the right side.
    # The cost is dy + (dx + (1 if (sx+sy)%2 == (tx+ty)%2 else 0)) // 2.
    # Wait, Sample 1: dx=3, dy=5, parity same. (3+1)//2 = 2. 5+2=7. Still not 5.
    # Let's re-read: "Each time he enters a tile, he pays a toll of 1."
    # Starting tile is free.
    # Sample 1: Start (5.5, 0.5). Tile is A_{5,0}. 5+0=5 (odd). 
    # Rule: i+j even => A_{i,j} and A_{i+1,j} are one tile.
    # For A_{5,0}, i=5, j=0. i+j=5 (odd). So A_{5,0} is NOT paired with A_{6,0}.
    # It is paired with A_{4,0} because 4+0=4 (even).
    # So tile is {A_{4,0}, A_{5,0}}.
    # Start is in tile {4,5} row 0.
    # Move left to 4.5: still in tile {4,5}. Toll 0.
    # Move up to 4.5, 1.5: enters tile in row 1. A_{4,1}. 4+1=5 (odd).
    # A_{4,1} is paired with A_{3,1} (3+1=4 even).
    # So he enters tile {3,4} row 1. Toll 1.
    # Move left to 3.5: still in tile {3,4}. Toll 0.
    # Move up to 3.5, 4.5: enters tiles in row 2, 3, 4. Toll 3.
    # In row 4, A_{3,4} is paired with A_{4,4} (4+4=8 even).
    # He is at 3.5, so he is in tile {3,4} row 4.
    # Move left to 2.5: enters tile {2,3} row 4 (2+4=6 even). Toll 1.
    # Move up to 2.5, 5.5: enters tile in row 5. A_{2,5}. 2+5=7 (odd).
    # A_{2,5} is paired with A_{1,5} (1+5=6 even).
    # He enters tile {1,2} row 5. Toll 1.
    # Total: 1 + 3 + 1 + 1 = 6. Still not 5.
    # Let's re-read: "Move left by 1. Pay a toll of 0. Move up by 1. Pay a toll of 1..."
    # The sample says: 0 + 1 + 0 + 3 + 0 + 1 = 5.
    # The last move "Move up by 1" enters the final tile.
    # The key is that he can move horizontally for free within a tile.
    # The cost is dy + (dx + (1 if (sx+sy)%2 == (tx+ty)%2 else 0)) // 2.
    # Let's check Sample 1 again: sx=5, sy=0, tx=2, ty=5.
    # (sx+sy)%2 = 1. (tx+ty)%2 = 7%2 = 1.
    # dx = 3. dy = 5.
    # Cost = 5 + (3 + 1)//2 = 7? No.
    # Wait, the sample says he moves left, then up, then left, then up...
    # He is alternating.
    # Each "up" move can be combined with a "left"