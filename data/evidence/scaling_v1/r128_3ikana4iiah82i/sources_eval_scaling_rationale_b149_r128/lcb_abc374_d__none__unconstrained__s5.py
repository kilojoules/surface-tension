import sys
import math
from itertools import permutations, product

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of endpoints
    segments = [
        ((int(input_data[3 + 4*i]), int(input_data[4 + 4*i])), 
         (int(input_data[5 + 4*i]), int(input_data[6 + 4*i])))
        for i in range(N)
    ]
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    # The total time spent printing (emitting laser) is constant regardless of order
    # Total print time = sum(length of each segment / T)
    print_time = sum(dist(s[0], s[1]) / T for s in segments)
    
    # We need to find the minimum travel time (not emitting laser)
    # There are N! permutations of segments and 2^N ways to choose the direction of each segment
    # Since N is small (<= 6), we can brute force all combinations
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # Generate all possible direction choices (0: start->end, 1: end->start)
    all_dirs = product([0, 1], repeat=N)
    
    # For a given permutation and direction set, calculate the travel distance
    # Travel distance = dist(0,0 to first start) + sum(dist(current end to next start))
    
    # To avoid loops, we use a comprehension to calculate the total travel distance for one configuration
    # We use a helper function to get the start and end points based on the direction
    get_pts = lambda seg_idx, direction: (
        (segments[seg_idx][0], segments[seg_idx][1]) if direction == 0 
        else (segments[seg_idx][1], segments[seg_idx][0])
    )

    # We calculate the travel distance for every permutation and every direction combination
    # Note: We use a nested comprehension. 
    # The outer layers iterate through permutations and direction bits.
    
    # To optimize, we can pre-calculate the travel distance for a specific sequence and direction
    # travel_dist = dist((0,0), first_start) + sum(dist(end_i, start_{i+1}))
    
    # Because we cannot use loops, we use a list comprehension to evaluate all possibilities
    # and then take the minimum.
    
    # We use a trick with zip to calculate the distance between consecutive segments
    # sequence: [(start1, end1), (start2, end2), ...]
    # travel: dist((0,0), start1) + dist(end1, start2) + dist(end2, start3) ...
    
    # Since we need to iterate over permutations and directions, and N is small,
    # we can combine them into one large comprehension.
    
    # To handle the "sum of distances between ends and starts", we map a function
    # over the zipped sequence of segments.
    
    # We define a function to calculate travel time for a specific order and direction
    def calc_travel(perm, dirs):
        # Create the sequence of (start, end) points
        seq = [get_pts(perm[i], dirs[i]) for i in range(N)]
        # Distance from origin to first start
        d0 = dist((0, 0), seq[0][0])
        # Distances between segments
        d_between = sum(dist(seq[i][1], seq[i+1][0]) for i in range(N-1))
        return (d0 + d_between) / S

    # We use a generator expression inside min() to find the minimum travel time
    # We iterate over all permutations and all possible direction combinations
    # Note: The directions are tied to the segments, so we can just use a bitmask or product
    # But the direction choice is independent of the permutation order.
    # Actually, the direction choice is relative to the segment itself.
    
    # Let's refine: for a fixed permutation, we want to choose directions to minimize travel.
    # However, the direction of segment i affects the travel to segment i+1.
    # This looks like DP, but with N=6, 2^6 * 6! is only 64 * 720 = 46,080.
    # This is small enough for a brute force comprehension.
    
    min_travel_time = min(
        calc_travel(p, d) 
        for p in permutations(range(N)) 
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()