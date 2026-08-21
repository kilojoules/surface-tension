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

    # Precompute lengths of segments (time spent emitting laser)
    # length / T
    seg_times = [math.sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) / T for s in segments]
    total_emit_time = sum(seg_times)

    # We need to find the optimal order of segments and the direction of each segment.
    # There are N! permutations and 2^N direction combinations.
    # For N=6, 720 * 64 = 46,080 combinations, which is feasible.
    
    # Each segment i has two endpoints: P_{i,0} and P_{i,1}.
    # If we print segment i from P_{i,0} to P_{i,1}, the laser ends at P_{i,1}.
    
    # Generate all permutations of segment indices
    perms = itertools.permutations(range(N))
    
    # Generate all possible direction assignments (0: start at P0, 1: start at P1)
    # For a segment i, if dir=0, it goes P_{i,0} -> P_{i,1}. If dir=1, P_{i,1} -> P_{i,0}.
    dirs = itertools.product([0, 1], repeat=N)
    
    # To avoid loops, we can't use 'for' but we can use map/min/comprehensions.
    # However, the constraint says "Return only Python source... Provide a complete, working solution."
    # Usually, "no for loops" is a specific constraint for certain challenges. 
    # If the prompt allows standard Python, loops are fine. 
    # But to be safe and "functional", I'll use comprehensions.
    
    # We want to minimize the travel time (distance / S).
    # The total time is total_emit_time + (total_travel_distance / S).
    
    # Let's define a helper to get the endpoints based on direction.
    # endpoints(i, direction) -> (start_point, end_point)
    get_pts = lambda i, d: (segments[i][0], segments[i][1]) if d == 0 else (segments[i][1], segments[i][0])
    
    # Distance function
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # For a fixed permutation and fixed directions, calculate travel distance.
    # Start at (0,0).
    # Travel: (0,0) -> start_1 -> end_1 -> start_2 -> end_2 ...
    
    # We can optimize by iterating over permutations and for each, 
    # finding the best directions using DP or exhaustive search.
    # Since N is small, exhaustive search for directions is fine.
    
    # To avoid 'for' loops entirely:
    # We use a nested comprehension.
    # The outer layer: permutations of segments.
    # The inner layer: all 2^N direction combinations.
    
    # Actually, for a fixed permutation, the choice of direction for segment i 
    # only depends on the end point of segment i-1 and the start point of segment i+1.
    # But since N is only 6, 2^N is tiny.
    
    ans_travel_dist = min(
        sum(
            dist(
                (0, 0) if step == 0 else get_pts(p[step-1], d[p[step-1]])[1],
                get_pts(p[step], d[p[step]])[0]
            )
            for step in range(N)
        )
        for p in perms
        for d in itertools.product([0, 1], repeat=N)
    )

    print(f"{total_emit_time + ans_travel_dist / S:.20f}")

if __name__ == "__main__":
    solve()