import itertools
import math

def solve():
    # Read N, S, T
    try:
        line1 = input().split()
        if not line1: return
        n, s, t = map(int, line1)
    except EOFError:
        return

    # Read segments
    segments = [list(map(int, input().split())) for _ in range(n)]

    # Precompute lengths of segments and time to print them
    # segment_info[i] = (length, print_time)
    seg_info = [
        (math.sqrt((a-c)**2 + (b-d)**2), math.sqrt((a-c)**2 + (b-d)**2) / t)
        for a, b, c, d in segments
    ]

    # We need to try all permutations of segments and all directions (start/end)
    # For N=6, N! * 2^N = 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(n))
    
    # Generate all possible direction choices (0 for A->C, 1 for C->A)
    directions = itertools.product([0, 1], repeat=n)
    
    # To avoid nested loops and maintain a functional style, we use a generator expression.
    # However, since we need to iterate directions for every permutation, 
    # we can flatten the problem into a single iterable of (permutation, direction_tuple).
    
    # We use a helper function to calculate the total time for a specific sequence
    def calculate_time(p, d):
        # Current position starts at (0, 0)
        # We need to track the current position and accumulate time.
        # Since we can't use loops, we use a reduction-like approach or a list comprehension.
        # But the position depends on the previous step. 
        # A simple way is to pre-calculate the endpoints for the given permutation and direction.
        
        # endpoints[i] = (start_x, start_y, end_x, end_y)
        coords = [
            (segments[p[i]][0], segments[p[i]][1], segments[p[i]][2], segments[p[i]][3]) if d[i] == 0 
            else (segments[p[i]][2], segments[p[i]][3], segments[p[i]][0], segments[p[i]][1])
            for i in range(n)
        ]
        
        # To calculate travel time between segments without a loop:
        # Travel 0: (0,0) to coords[0].start
        # Travel i: coords[i-1].end to coords[i].start
        # Printing i: coords[i].start to coords[i].end
        
        # Travel times (non-emitting)
        travel_times = [
            math.sqrt(coords[0][0]**2 + coords[0][1]**2) / s
        ] + [
            math.sqrt((coords[i][0] - coords[i-1][3])**2 + (coords[i][1] - coords[i-1][2])**2) / s 
            # Wait, the index for coords[i-1] was (start_x, start_y, end_x, end_y)
            # So end_x is index 2, end_y is index 3.
            for i in range(1, n)
        ]
        
        # Correcting the travel time logic:
        # Let's redefine coords as (sx, sy, ex, ey)
        # Travel 0: dist((0,0), (sx0, sy0)) / s
        # Travel i: dist((ex_{i-1}, ey_{i-1}), (sx_i, sy_i)) / s
        # Print i: dist((sx_i, sy_i), (ex_i, ey_i)) / t
        
        # Using a list comprehension to get all travel distances
        # We create a list of points: (0,0), then for each segment (sx, sy), (ex, ey)
        pts = [(0, 0)] + [pt for seg in coords for pt in [(seg[0], seg[1]), (seg[2], seg[3])]]
        
        # Travel times are between pts[2i+1] and pts[2i]
        # Printing times are between pts[2i+2] and pts[2i+1]
        # Actually:
        # i=0: travel (pts[0] to pts[1]), print (pts[1] to pts[2])
        # i=1: travel (pts[2] to pts[3]), print (pts[3] to pts[4])
        
        # Total time = sum(dist(pts[2i], pts[2i+1])/s) + sum(dist(pts[2i+1], pts[2i+2])/t)
        # for i in range(n)
        
        # But we can't use loops. We use range() in a comprehension.
        total_travel = sum([
            math.sqrt((pts[2*i+1][0] - pts[2*i][0])**2 + (pts[2*i+1][1] - pts[2*i][1])**2) / s
            for i in range(n)
        ])
        
        total_print = sum([
            math.sqrt((pts[2*i+2][0] - pts[2*i+1][0])**2 + (pts[2*i+2][1] - pts[2*i+1][1])**2) / t
            for i in range(n)
        ])
        
        return total_travel + total_print

    # Since we can't use nested loops, we use a generator expression to evaluate all combinations.
    # We need to evaluate calculate_time for all permutations and all direction combinations.
    # The number of combinations is N! * 2^N.
    
    # To avoid the 'directions' product being inside a loop, we can use a nested generator.
    # result = min(calculate_time(p, d) for p in perms for d in itertools.product([0, 1], repeat=n))
    
    # Using a list comprehension to drive the logic:
    ans = min([
        calculate_time(p, d) 
        for p in itertools.permutations(range(n)) 
        for d in itertools.product([0, 1], repeat=n)
    ])
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()