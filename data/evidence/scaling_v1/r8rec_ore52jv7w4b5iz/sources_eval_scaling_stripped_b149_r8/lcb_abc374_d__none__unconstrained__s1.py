import itertools
import math
from functools import reduce

def solve():
    # Read input from stdin
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        segments.append((int(input_data[idx]), int(input_data[idx+1]), 
                         int(input_data[idx+2]), int(input_data[idx+3])))
        idx += 4

    # Pre-calculate lengths of segments and distances between all endpoints
    # Each segment i has endpoints P_{i,0} and P_{i,1}
    coords = [((s[0], s[1]), (s[2], s[3])) for s in segments]
    seg_lengths = [math.sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) for s in segments]
    
    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to find the best permutation of segments and the best direction for each
    # Since N is small (<= 6), we can iterate through all permutations and all 2^N directions.
    # A state can be represented as (current_permutation, current_directions)
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(N))
    # Generate all possible direction combinations (0: P0->P1, 1: P1->P0)
    dirs = itertools.product([0, 1], repeat=N)
    
    # To avoid loops, we can't use a simple generator expression inside min() if we 
    # need to avoid 'for' loops. We use map and reduce or list comprehensions.
    
    # For a fixed permutation and fixed directions, calculate total time.
    # Let the sequence of segments be s_0, s_1, ..., s_{N-1}
    # Let the chosen directions be d_0, d_1, ..., d_{N-1}
    # Start point is (0,0).
    
    def calculate_time(p, d):
        # Sequence of points: Start -> Start_of_Seg0 -> End_of_Seg0 -> Start_of_Seg1 ...
        # Points are:
        # Pt 0: (0,0)
        # Pt 1: coords[p[0]][d[0]]
        # Pt 2: coords[p[0]][1 - d[0]]
        # Pt 3: coords[p[1]][d[1]] ...
        
        # We can construct the sequence of points visited
        pts = [(0, 0)] + [
            (coords[p[i]][d[i]], coords[p[i]][1 - d[i]]) 
            for i in range(N)
        ]
        # Flatten the list of tuples
        flat_pts = reduce(lambda acc, x: acc + [x], pts, [])
        
        # Calculate travel times:
        # Even indices i (0, 2, ...): travel from flat_pts[i] to flat_pts[i+1] (non-emitting)
        # Odd indices i (1, 3, ...): travel from flat_pts[i] to flat_pts[i+1] (emitting)
        # Note: the first segment is special. The move from (0,0) to the first endpoint is non-emitting.
        
        # The segments are:
        # (flat_pts[0], flat_pts[1]) -> Move (S)
        # (flat_pts[1], flat_pts[2]) -> Print (T)
        # (flat_pts[2], flat_pts[3]) -> Move (S)
        # ...
        
        # Use a list comprehension to calculate costs for each leg
        costs = [
            (dist(flat_pts[i], flat_pts[i+1]) / S) if i % 2 == 0 
            else (dist(flat_pts[i], flat_pts[i+1]) / T)
            for i in range(2 * N)
        ]
        return sum(costs)

    # We need to evaluate calculate_time for all permutations and all direction combinations.
    # Since we can't use for-loops, we use nested comprehensions.
    # Total complexity: N! * 2^N * N
    
    # To avoid the 'for' keyword entirely, we use map/comprehensions.
    # We use a list comprehension to iterate over permutations and directions.
    ans = min([
        calculate_time(p, d) 
        for p in perms 
        for d in itertools.product([0, 1], repeat=N)
    ])
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()