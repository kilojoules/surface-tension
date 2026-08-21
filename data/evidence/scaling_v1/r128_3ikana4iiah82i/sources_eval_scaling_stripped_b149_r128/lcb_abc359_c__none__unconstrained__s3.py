import sys
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    sx, sy, tx, ty = map(int, input_data)

    # The tiling rule: 
    # If i+j is even, A_{i,j} and A_{i+1,j} are one tile.
    # This means for a fixed j, the tiles are pairs (0,1), (2,3), etc. if j is even.
    # If j is odd, the tiles are pairs (1,2), (3,4), etc.
    # Essentially, the tile boundaries in x are at x = k where k % 2 == (j % 2).
    
    # To move from (sx, sy) to (tx, ty), we can move in x then y, or y then x.
    # However, we can optimize by moving to a coordinate that minimizes the toll.
    # The cost to move from sy to ty is simply abs(sy - ty) because every 
    # vertical move of 1 unit enters a new tile (since tiles are 2x1).
    # The cost to move from sx to tx depends on the parity of the current y.
    
    # Let's evaluate the cost of moving horizontally at height y.
    # A move from sx to tx at height y:
    # We pay a toll if we cross a boundary x = k where k % 2 == (y % 2).
    # The number of such boundaries between sx and tx is:
    # count(k in (min(sx, tx), max(sx, tx)) such that k % 2 == (y % 2))
    
    def get_horizontal_cost(x1, x2, y):
        low, high = min(x1, x2), max(x1, x2)
        # We want to count k such that low < k < high and k % 2 == y % 2.
        # This is equivalent to counting k in [low + 1, high - 1] with k % 2 == y % 2.
        if low + 1 > high - 1:
            return 0
        
        # Number of integers in [1, N] with parity p is (N + (1 if p==1 else 0)) // 2
        # But it's easier to use: count of k <= N with k % 2 == p is (N + 1) // 2 if p=1 else N // 2
        # Wait, the standard formula for count of k in [0, N] with k % 2 == p:
        # If p=0: (N // 2) + 1
        # If p=1: (N + 1) // 2
        
        def count_upto(n, p):
            if n < 0: return 0
            return (n // 2) + (1 if p == 0 else 0) if p == 0 else (n + 1) // 2
        
        # For p=0, count_upto(n, 0) is 0, 2, 4... -> n//2 + 1
        # For p=1, count_upto(n, 1) is 1, 3, 5... -> (n+1)//2
        # Let's redefine to avoid the 0-index confusion and use a simpler logic:
        # The number of k in [1, N] with k % 2 == p is:
        # if p == 1: (N + 1) // 2
        # if p == 0: N // 2
        
        def count_range(n, p):
            if n <= 0: return 0
            return (n + 1) // 2 if p == 1 else n // 2

        # We need k in [low + 1, high - 1].
        # This is count_range(high - 1, y % 2) - count_range(low, y % 2).
        # However, the parity logic depends on whether the coordinate is shifted.
        # Let's use: the number of k in (low, high) with k % 2 == y % 2.
        # This is equivalent to counting k in [0, high-1] minus k in [0, low].
        # Let f(n, p) be the number of k in {1, ..., n} such that k % 2 == p.
        # f(n, 1) = (n + 1) // 2
        # f(n, 0) = n // 2
        
        p = y % 2
        return count_range(high - 1, p) - count_range(low, p)

    # The total cost is abs(sy - ty) + horizontal_cost.
    # We can choose to move horizontally at sy or at ty.
    # Any other y would only increase the vertical cost without decreasing horizontal cost.
    
    cost_at_sy = abs(sy - ty) + get_horizontal_cost(sx, tx, sy)
    cost_at_ty = abs(sy - ty) + get_horizontal_cost(sx, tx, ty)
    
    print(min(cost_at_sy, cost_at_ty))

if __name__ == "__main__":
    solve()