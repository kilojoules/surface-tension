# Pilot Study: Binding Coding Constraints

**Goal:** find (benchmark, constraint) pairs with high `binding_score = performance_drop × compliance_rate`.

Model: `gemini-2.5-flash` (spec called for 2.0-flash, but only 2.5-flash is available via the CLI as of generation date).
Raw data: `pilot_v3_raw.csv` (342 rows).

## Data sufficiency
## Success-criteria check
Pilot is informative if some pair has `drop_overall ≥ 0.15` (unbiased binding pressure).

**1 pair(s) cleared the threshold:**
- `lcb_medium` × `no_loops_no_recursion`: drop_overall=0.1765, compliance=0.7794

## Full ranking

Sorted by `drop_overall` (unbiased — counts non-compliant constrained samples as failures).
`drop_among_compliant` is the spec metric, kept for reference but biased upward when compliance < 1.

| benchmark | constraint | n_unc | n_con | n_compl | rej_unc | rej_con | pass_unc | **pass_overall** | **drop_overall** | compliance | pass_compl | drop_compl |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lcb_medium | no_loops_no_recursion | 68 | 68 | 53 | 0.6023 | 0.6023 | 0.8676 | **0.6912** | **0.1765** | 0.7794 | 0.8868 | -0.0191 |

## Top 3 examples

### 1. `lcb_medium` × `no_loops_no_recursion` (drop_overall=0.1765)

- pass_unconstrained: **0.8676**, pass_overall: **0.6912**
- compliance_rate: **0.7794**, drop_among_compliant (biased): -0.0191

**Example problem:** `lcb/abc359_c`

**Unconstrained solution:**
```python
import sys

def solve():
    """
    This function reads input, calculates the minimum toll, and prints the result.
    """
    try:
        s_x_str, s_y_str = sys.stdin.readline().split()
        t_x_str, t_y_str = sys.stdin.readline().split()
        s_x = int(s_x_str)
        s_y = int(s_y_str)
        t_x = int(t_x_str)
        t_y = int(t_y_str)
    except (IOError, ValueError):
        # Handle potential empty lines or invalid input at the end of the file
        return

    # Function to find the starting x-coordinate 'k' of the tile containing the square A_{i,j}.
    # A tile is a 2x1 rectangle [k, k+2] x [j, j+1] where k+j is even.
    # For a square A_{i,j}, we need to find k such that k <= i <= k+1 and k+j is even.
    def get_k(i, j):
        if (i + j) % 2 == 0:
            return i
        else:
            return i - 1

    # Get the k-coordinates for the start and target tiles.
    k_s = get_k(s_x, s_y)
    k_t = get_k(t_x, t_y)

    # If start and target points are in the same tile, the toll is 0.
    # This happens if they are in the same row of squares (s_y == t_y)
    # and belong to the same tile (k_s == k_t).
    if s_y == t_y and k_s == k_t:
        print(0)
        return

    # Calculate the difference in y-coordinates and k-coordinates.
    dy = abs(s_y - t_y)
    dk = abs(k_s - k_t)

    # The minimum toll can be found by considering the costs of vertical and horizontal movements.
    # A vertical move of 1 unit (e.g., from y to y+1) always costs 1 toll.
    # A horizontal move changes tile every 2 units of x.
    #
    # Let's analyze the costs in a transformed coordinate system where a tile is a node in a graph.
    # Moving one step in y costs 1.
    # Moving one step in k (which is a shift of 2 in x) costs 1.
    # A diagonal move (1 in y, 1 in k) also costs 1.
    #
    # From a tile at (k, y), we can reach:
    # - (k, y+1) or (k, y-1): cost 1 (if k has correct parity for new y)
    # - (k+2, y) or (k-2, y): cost 1
    # - (k+1, y+1) or (k-1, y+1) etc: cost 1
    #
    # The number of vertical steps is `dy`.
    # The number of horizontal steps (in k-space) is `dk`.
    # Each vertical step can accommodate one horizontal k-step without extra cost.
    # e.g. moving from tile (k, y) to (k+1, y+1) costs 1.
    # So, `dy` steps can cover `dy` units of `dk`.
    # The remaining horizontal distance is `dk - dy`.
    # This remaining distance must be covered by horizontal-only moves,
    # where each move of 2 in k costs 2 tolls (e.g., (k,y)->(k+2,y)->(k+4,y)).
    # A move of `dk` in k-space costs `dk/2` tolls.
    # So remaining `dk - dy` costs `(dk - dy) / 2`.
    # Total cost = `dy + (dk - dy) / 2 = (dy + dk) / 2` if `dk > dy`.
    # If `dy >= dk`, all horizontal movement can be absorbed by diagonal moves, so cost is `dy`.
    # This is equivalent to max(dy, (dk+dy)/2).
    
    # Let's verify the cost logic.
    # If only moving vertically (dk=0), cost is dy. Correct.
    # If only moving horizontally (dy=0), cost is dk/2. Correct, since each horizontal tile step covers 2 units of x.
    # If dy=dk=d, path is purely diagonal, cost is d.
    # Formula gives max(d, (d+d)/2) = max(d,d) = d. Correct.

    # We need to calculate `(dk + dy) // 2` using integer division.
    # If `dk < dy`, we can absorb all horizontal moves into vertical moves.
    # Cost is `dy`.
    # `max(dy, (dk+dy)//2)` does not work if `dk` and `dy` have different parity.
    # Example: dy=2, dk=1. max(2, (1+2)//2) = max(2,1) = 2.
    # Path: (k,y) -> (k+1, y+1) -> (k+1, y+2). Cost 2. Correct.
    # Example: dy=1, dk=2. max(1, (2+1)//2) = max(1,1)=1. Incorrect.
    # Path: (k,y) -> (k+1,y+1) -> (k+2,y+1). Cost 2.
    # Correct cost is `dy + (dk-dy)//2` if dk>dy. Here `1 + (2-1)//2 = 1`.
    
    ans = 0
    if dk <= dy:
        # All horizontal movement can be absorbed into diagonal moves.
        # However, we must account for parity.
        # If (dk-dy) is odd, we need an extra horizontal move.
        if (dy - dk) % 2 == 1:
            # e.g., dy=2, dk=1. dy-dk=1 (odd).
            # (k,y)->(k+1,y+1)->(k+1,y+2). k needs to change by 1. y by 2.
            # (dy-dk)%2==1 means we end up at a y-level with wrong k-parity.
            # We need to shift one tile over.
            # This happens at the end of the vertical moves.
            # It seems the number of purely horizontal steps needed is `(dk-dy)//2`
            # and the number of purely vertical steps is `(dy-dk)//2`.
            # Total cost is number of diagonal steps + horiz + vert.
            # num_diag = (dy+dk - abs(dy-dk)) / 2 = min(dy,dk)
            # cost = min(dy,dk) + abs(dy-dk) is not right.
            
            # Let's re-evaluate the cost of moves:
            # vertical step: (k,y) -> (k-1,y+1) or (k+1,y+1). dy->-1, dk->-1. cost 1.
            # horizontal step: (k,y) -> (k+2,y). dy->0, dk->-2. cost 1.
            # To achieve (dk, dy) difference:
            # Let `a` be number of diag moves, `b` be horiz moves, `c` be vert moves.
            # Minimize `a+b+c`.
            # dy = a+c, dk = a*1 + b*2.
            # from dy=a+c => c=dy-a. Since c>=0, a<=dy.
            # from dk=a+2b => 2b=dk-a. Since b>=0, a<=dk. Also dk-a must be even.
            # so, a <= min(dy,dk) and a=dk (mod 2).
            # cost = a+b+c = a + (dk-a)/2 + dy-a = dy + (dk-a)/2.
            # To minimize cost, we need to maximize `a`.
            # So, `a` is the largest integer <= min(dy,dk) with same parity as `dk`.
            a = min(dy, dk)
            if a % 2 != dk % 2:
                a -= 1
            
            if a < 0: # a must be non-negative
                b = dk/2
                c = dy
                ans = b+c
            else:
                b = (dk - a) / 2
                c = dy - a
                ans = a + b + c
        else: # (dy-dk)%2 == 0
            ans = dy
    else: # dk > dy
        # Not enough vertical moves to absorb horizontal ones.
        # We make `dy` diagonal moves, covering `dy` of `dk`.
        # Remaining horizontal distance `dk - dy`.
        # This must be covered by horizontal moves, each costing 1 and covering 2 k-units.
        # Cost is `dy` (for the diagonal part) + `(dk - dy) / 2`.
        ans = dy + (dk - dy) // 2

    print(int(ans))

solve()
```

**Constrained-compliant solution:**
```python
import math
import sys

def solve():
    """
    Solves the Tiling problem by calculating the minimum toll.

    The solution is based on the following observations:
    1. A diagonal move (e.g., one step right and one step up) can always be
       arranged to have a total cost of 0. This is because the two possible
       paths (Right then Up, or Up then Right) have complementary costs;
       if one path costs 2, the other costs 0.
    2. An anti-diagonal move (e.g., one step right and one step down) always
       costs 1, regardless of the path taken.

    Based on this, the optimal strategy is to maximize the number of free
    diagonal moves. The number of diagonal moves is min(|dx|, |dy|), where
    dx and dy are the total horizontal and vertical distances.

    After making all possible diagonal moves, the remaining travel is a
    straight line, either purely horizontal or purely vertical. The length
    of this remaining segment is |dx| - |dy||.

    The cost of this final straight-line move is calculated based on the
    parity of the coordinates, as the cost of crossing a grid line alternates.
    - Cost of horizontal move from (i,j) to (i+1,j) is 1 if (i+1)+j is even.
    - Cost of vertical move from (i,j) to (i,j+1) is 1 if i+(j+1) is odd.
    The total cost for N steps is either floor(N/2) or ceil(N/2) depending
    on the starting parity.

    The entire calculation is performed without loops or recursion, using
    arithmetic and conditional expressions.
    """
    sx, sy = map(int, sys.stdin.readline().split())
    tx, ty = map(int, sys.stdin.readline().split())

    # Ensure sx <= tx for consistent calculations
    sx, sy, tx, ty = (sx, sy, tx, ty) if sx <= tx else (tx, ty, sx, sy)
    
    dx = tx - sx
    dy = abs(ty - sy)
    
    # Number of free diagonal moves
    k = min(dx, dy)
    
    # The remaining move is a straight line of length N
    N = dx - k
    
    # Starting point of the final straight-line move.
    # We've made k steps of size (1, sign(ty-sy)) from (sx, sy)
    ix = sx + k
    iy = sy + k * (1 if ty >= sy else -1)

    # Cost calculation for the straight horizontal move of length N
    # The cost of moving from (i,j) to (i+1,j) is 1 if (i+1)+j is even, i.e., i+j is odd.
    # P is 1 if the first step has a cost, 0 otherwise.
    P = (ix + iy) % 2 == 1
    # For a move of length N, if the first step costs 1, the costs are 1,0,1,0...
    # The number of '1's is ceil(N/2).
    # If the first step costs 0, the costs are 0,1,0,1...
    # The number of '1's is floor(N/2).
    cost = math.ceil(N / 2) * P + math.floor(N / 2) * (1 - P)

    # Now consider the path from (sx,sy) to (sx,ty), then (sx,ty) to (tx,ty)
    # The logic is symmetric. We calculate the cost for this path and take the minimum.
    # Vertical move first, of length dy. Then horizontal of length dx.
    
    # Cost for vertical move of length dy from (sx, sy)
    # The cost of moving from (i,j) to (i,j+1) is 1 if i+j+1 is odd, i.e., i+j is even.
    # Start point is (sx, min(sy,ty)). Length is dy.
    v_start_y = min(sy, ty)
    # P_v is 1 if the first vertical step has a cost.
    P_v = (sx + v_start_y) % 2 == 0
    cost_v = math.ceil(dy / 2) * P_v + math.floor(dy / 2) * (1 - P_v)
    
    # Cost for horizontal move of length dx from (sx, ty)
    # P_h is 1 if the first horizontal step has a cost.
    P_h = (sx + ty) % 2 == 1
    cost_h = math.ceil(dx / 2) * P_h + math.floor(dx / 2) * (1 - P_h)

    print(min(cost, cost_v + cost_h))

solve()
```
