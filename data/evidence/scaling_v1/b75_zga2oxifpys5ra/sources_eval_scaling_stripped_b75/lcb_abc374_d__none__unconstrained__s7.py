import itertools
import math

def solve():
    # Read N, S, T
    line1 = input().split()
    if not line1: return
    n, s, t = map(int, line1)
    
    # Read line segments
    segments = [list(map(int, input().split())) for _ in range(n)]
    
    # Pre-calculate lengths of each segment
    # length = sqrt((Cx-Ax)^2 + (Dy-By)^2)
    # Time to print = length / T
    seg_lengths = [math.sqrt((seg[2]-seg[0])**2 + (seg[3]-seg[1])**2) for seg in segments]
    
    # Each segment i has two endpoints: P_{i,0}=(Ai, Bi) and P_{i,1}=(Ci, Di)
    # We need to decide the order of segments and the direction of each segment.
    # There are N! permutations of segments and 2^N choices of directions.
    
    # Let's represent a state as (segment_index, endpoint_index)
    # endpoint_index 0 means we finished at (Ci, Di), 1 means we finished at (Ai, Bi)
    # Wait, that's confusing. Let's say:
    # Direction 0: Start at (Ai, Bi), end at (Ci, Di)
    # Direction 1: Start at (Ci, Di), end at (Ai, Bi)
    
    def get_coords(seg_idx, direction, is_start):
        seg = segments[seg_idx]
        # direction 0: start=(A,B), end=(C,D)
        # direction 1: start=(C,D), end=(A,B)
        if direction == 0:
            return (seg[0], seg[1]) if is_start else (seg[2], seg[3])
        else:
            return (seg[2], seg[3]) if is_start else (seg[0], seg[1])

    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # Generate all permutations of segment indices
    perms = itertools.permutations(range(n))
    # Generate all combinations of directions (0 or 1)
    dirs = itertools.product([0, 1], repeat=n)
    
    # We want to minimize:
    # Total Time = Sum(seg_lengths / T) + Sum(dist(end_prev, start_curr) / S)
    # The first movement is from (0,0) to start_0.
    
    # Since we can't use loops, we use a generator expression inside min()
    # We iterate over all permutations and all direction combinations.
    
    # To avoid nested loops, we can use a helper logic to calculate total travel time
    # for a specific permutation and direction set.
    
    def calculate_travel_time(p, d):
        # Starting point
        curr_pos = (0, 0)
        travel_dist = 0
        # For each segment in the permutation
        # We need the start and end coordinates based on the direction
        # We use a list comprehension to build the sequence of (start, end) pairs
        stops = [ (get_coords(p[i], d[p[i]], True), get_coords(p[i], d[p[i]], False)) 
                  for i in range(n) ]
        
        # The travel distance is dist((0,0), start_0) + dist(end_0, start_1) + ...
        # We can use a trick with zip to calculate distances between end_i and start_{i+1}
        starts = [s[0] for s in stops]
        ends = [s[1] for s in stops]
        
        # Initial jump
        initial_dist = dist((0,0), starts[0])
        # Intermediate jumps
        intermediate_dist = sum(dist(ends[i], starts[i+1]) for i in range(n-1))
        
        return (initial_dist + intermediate_dist) / s

    # We need to map the directions to the permutation. 
    # The 'dirs' product generates tuples of length N. 
    # If we use a permutation p, the direction of segment p[i] is d[i].
    
    # Corrected logic for directions:
    # For a permutation p, there are 2^N ways to orient the segments.
    # Let d be a tuple of length N where d[i] is the direction of the i-th segment in the permutation.
    
    total_print_time = sum(seg_lengths) / t
    
    # We use a generator to find the minimum travel time across all permutations and directions.
    # Note: N is small (up to 6), so N! * 2^N = 720 * 64 = 46,080, which is well within limits.
    
    min_travel_time = min(
        sum(
            dist(
                (0, 0) if i == 0 else get_coords(p[i-1], d[i-1], False),
                get_coords(p[i], d[i], True)
            )
            for i in range(n)
        ) / s
        for p in itertools.permutations(range(n))
        for d in itertools.product([0, 1], repeat=n)
    )

    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()