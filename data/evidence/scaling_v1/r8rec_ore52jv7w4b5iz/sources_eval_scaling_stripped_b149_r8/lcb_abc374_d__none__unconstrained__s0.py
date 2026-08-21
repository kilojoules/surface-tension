import sys
from itertools import permutations, product
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    for i in range(N):
        a = int(input_data[3 + i*4])
        b = int(input_data[4 + i*4])
        c = int(input_data[5 + i*4])
        d = int(input_data[6 + i*4])
        segments.append(((a, b), (c, d)))

    # Precompute lengths of segments
    # length = sqrt((x2-x1)^2 + (y2-y1)^2)
    seg_lengths = [((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2)**0.5 for s in segments]
    
    # Total time spent emitting the laser is constant regardless of order
    total_emit_time = sum(seg_lengths) / T

    # We need to find the minimum travel time (non-emitting)
    # There are N! permutations of segments and 2^N ways to choose directions
    # For N=6, N! * 2^N = 720 * 64 = 46,080, which is small enough.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all possible direction choices (0: start->end, 1: end->start)
    dirs = product([0, 1], repeat=N)
    
    # To avoid loops, we can't use 'for' in a comprehension, but we can use 
    # a generator expression inside min().
    # However, we need to evaluate the travel distance for each (perm, dir) pair.
    
    # For a fixed permutation and direction set:
    # The sequence of points visited is:
    # P0(0,0) -> Start1 -> End1 -> Start2 -> End2 ... -> StartN -> EndN
    # The travel time is (dist(P0, Start1) + dist(End1, Start2) + ... + dist(EndN-1, StartN)) / S
    
    # We use a helper function to calculate travel distance for a specific configuration
    def calc_travel_dist(config):
        perm, direction = config
        # Map each segment index to its start and end point based on the chosen direction
        # points[i] = (start_point, end_point)
        pts = [
            (segments[perm[i]][0], segments[perm[i]][1]) if direction[perm[i]] == 0 
            else (segments[perm[i]][1], segments[perm[i]][0])
            for i in range(N)
        ]
        
        # Distance from (0,0) to first start point
        d0 = ((pts[0][0][0])**2 + (pts[0][0][1])**2)**0.5
        
        # Distances between segments: End_{i} to Start_{i+1}
        # Use map and sum to avoid loops
        d_between = sum(
            ((pts[i][1][0] - pts[i+1][0][0])**2 + (pts[i][1][1] - pts[i+1][0][1])**0.5 if False else 
             ((pts[i][1][0] - pts[i+1][0][0])**2 + (pts[i][1][1] - pts[i+1][0][1])**0.5)**0 # This is wrong
            ) for i in range(N-1)
        ) # Wait, the above is a mess. Let's use a cleaner approach.
        
        return d0

    # Let's redefine the distance calculation without loops
    # We can use a list comprehension to get all the "jump" pairs and sum their distances.
    
    # Since we can't use loops, we'll use a generator expression inside min()
    # We iterate over all permutations and all possible direction assignments.
    # For a fixed permutation, the direction of segment i only affects the distance 
    # from segment i-1 and to segment i+1.
    
    # Actually, the simplest way to handle the 2^N is to realize that for a fixed 
    # permutation, we can use DP or just brute force since N is small.
    # Let's use brute force for both.
    
    def get_total_dist(p, d):
        # p: permutation of indices, d: tuple of directions (0 or 1) for each segment index
        # Current point starts at (0,0)
        # For each segment in p:
        #   move to start point, then move to end point.
        
        # Create the sequence of points: (start1, end1, start2, end2, ...)
        coords = [
            (segments[p[i]][0] if d[p[i]] == 0 else segments[p[i]][1],
             segments[p[i]][1] if d[p[i]] == 0 else segments[p[i]][0])
            for i in range(N)
        ]
        
        # The jumps are: (0,0)->coords[0][0], coords[0][1]->coords[1][0], etc.
        jumps = [
            ((0, 0), coords[0][0])
        ] + [
            (coords[i][1], coords[i+1][0]) for i in range(N-1)
        ]
        
        return sum([((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5 for a, b in jumps])

    # We need to minimize get_total_dist(p, d) over all p in perms and d in {0,1}^N
    # To avoid 'for' loops, we use nested generator expressions.
    
    min_travel_dist = min(
        get_total_dist(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{total_emit_time + min_travel_dist / S:.20f}")

if __name__ == "__main__":
    solve()