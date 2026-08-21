import itertools
import math
import sys

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        c = int(input_data[idx+2])
        d = int(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precompute lengths of segments (time taken to print)
    # length / T
    seg_times = [math.sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) / T for s in segments]
    
    # Helper to calculate distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by 0 (start -> end) or 1 (end -> start).
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(N))
    # Generate all possible direction combinations (2^N)
    dirs = itertools.product([0, 1], repeat=N)
    
    # To avoid nested loops and maintain a functional approach, we use a generator expression.
    # For a fixed permutation and direction set:
    # The total time is:
    # sum(printing_times) + sum(travel_times)
    # travel_time_0: (0,0) to first endpoint
    # travel_time_i: end of segment i to start of segment i+1
    
    # We can't use 'for' loops, so we use map/sum/generator expressions.
    # Since we need to iterate over both perms and dirs, we use a nested generator.
    
    # Total printing time is constant regardless of order
    total_print_time = sum(seg_times)
    
    # We seek the minimum travel time.
    # For a given permutation 'p' and direction 'd':
    # Points visited: Start(0,0) -> P_{p0, d0_start} -> P_{p0, d0_end} -> P_{p1, d1_start} ...
    
    # Let's define a function to get the start and end points based on direction
    def get_endpoints(seg_idx, direction):
        seg = segments[seg_idx]
        return (seg[0], seg[1]) if direction == 0 else (seg[1], seg[0])

    # We use a generator to evaluate all combinations of permutations and directions.
    # We use a list comprehension inside min() to find the minimum travel time.
    # To avoid 'for' loops, we use itertools.product to combine perms and dirs.
    
    min_travel_time = min(
        sum(
            # Distance from current_end to next_start / S
            # We construct a list of (current_end, next_start) pairs.
            # The first pair is ((0,0), first_start).
            # The subsequent pairs are (end_i, start_{i+1}).
            [
                dist(
                    (0, 0) if i == 0 else get_endpoints(p[i-1], d[i-1])[1],
                    get_endpoints(p[i], d[i])[0]
                ) / S
                for i in range(N)
            ]
        )
        for p, d in itertools.product(perms, dirs)
    )

    # The result is total printing time + minimum travel time.
    # However, the logic above calculates travel time to the START of each segment.
    # Wait, the problem says: "move the laser position to one of the endpoints... then move... to the other".
    # So: (0,0) --S--> Start1 --T--> End1 --S--> Start2 --T--> End2 ...
    # My logic: travel to Start1, print to End1, travel to Start2, print to End2.
    # This matches the requirement.
    
    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()