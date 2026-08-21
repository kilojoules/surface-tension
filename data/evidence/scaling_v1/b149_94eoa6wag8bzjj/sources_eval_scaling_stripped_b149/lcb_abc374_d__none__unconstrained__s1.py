import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of points
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate printing time for each segment
    # print_times[i] is the time taken to print segment i
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments and all directions (start/end)
    # For a fixed permutation of segments, we can use DP or a simple reduce to find 
    # the min time. However, since N is very small (<= 6), we can just iterate 
    # through all 2^N direction combinations for each permutation.
    
    # To avoid explicit loops, we use map/comprehensions.
    # Each state in the direction bitmask: 0 means start->end, 1 means end->start.
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    
    # For a specific permutation and a specific direction mask, calculate total time.
    # mask is an integer from 0 to 2^N - 1.
    # For segment i in perm:
    #   if (mask >> i) & 1 == 0: entry = s[0], exit = s[1]
    #   else: entry = s[1], exit = s[0]
    
    def calc_time(perm, mask):
        # Create the sequence of (entry, exit) points based on the mask
        path = [
            (segments[i][1], segments[i][0]) if (mask & (1 << i)) else 
            (segments[i][0], segments[i][1])
            for i in perm
        ]
        
        # Calculate travel times between segments
        # Start point is (0, 0)
        # Travel 1: (0,0) -> path[0][0]
        # Travel 2: path[0][1] -> path[1][0] ...
        
        # We use a list of points to calculate distances
        # Points: Start, Entry1, Exit1, Entry2, Exit2 ...
        coords = [(0, 0)] + [pt for seg in path for pt in seg]
        
        # Travel times (non-emitting): dist(coords[i], coords[i+1]) / S for even i
        # Printing times (emitting): dist(coords[i], coords[i+1]) / T for odd i
        # But we already have print_times. Let's just sum travel times.
        
        travel_time = sum(
            dist(coords[i], coords[i+1]) / S 
            for i in range(0, len(coords) - 1, 2)
        )
        
        return travel_time + sum(print_times)

    # We use a nested comprehension to find the minimum across all permutations and masks.
    # Since N <= 6, total iterations = 6! * 2^6 = 720 * 64 = 46,080 (well within limits).
    ans = min(
        calc_time(p, m)
        for p in all_perms
        for m in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()