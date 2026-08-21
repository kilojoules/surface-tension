import sys
from itertools import permutations, product
from math import sqrt

def solve():
    # Read input
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

    # Precompute lengths of segments (time taken to print)
    # length_i = sqrt((A_i-C_i)^2 + (B_i-D_i)^2)
    # print_time_i = length_i / T
    seg_times = [sqrt((s[0]-s[2])**2 + (s[1]-s[3])**2) / T for s in segments]
    
    # We need to try all permutations of segments and all possible directions for each segment.
    # A direction is represented by 0 (start -> end) or 1 (end -> start).
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all possible direction combinations (2^N)
    dirs = product([0, 1], repeat=N)
    
    # To avoid nested loops and use a generator expression, we combine perms and dirs.
    # However, since we can't use loops, we'll use a nested comprehension/generator.
    # For each permutation and each direction set, calculate the total travel time.
    
    # We define a helper to get the coordinates of the endpoints based on direction.
    # endpoints[seg_idx][direction] = (x, y)
    endpoints = [((s[0], s[1]), (s[2], s[3])) for s in segments]
    
    # The total time is:
    # Sum of all print_times + Sum of travel times between segments.
    # Travel time = dist(current_pos, next_start_pos) / S
    
    # We use a generator expression to evaluate all combinations.
    # For a fixed permutation 'p' and direction 'd':
    # The sequence of points visited is:
    # Start (0,0) -> Start of seg p[0] -> End of seg p[0] -> Start of seg p[1] ...
    
    # Let's define the points for a specific permutation and direction set.
    # For segment i in permutation p with direction d_i:
    # Start point: endpoints[p[i]][d_i]
    # End point: endpoints[p[i]][1 - d_i]
    
    # Total time = sum(seg_times) + 
    #              dist((0,0), start_0)/S + 
    #              sum(dist(end_{i}, start_{i+1})/S for i in range(N-1))

    # Since we cannot use loops, we use map/sum/generator expressions.
    # We'll iterate over all permutations and all direction combinations.
    
    # To handle the "all directions" part without a loop, we can't use 'product' 
    # inside a comprehension if it depends on the permutation. 
    # Actually, the direction choice for segment i is independent of the permutation order,
    # but the travel distance depends on both.
    
    # Let's redefine: for each segment, we choose which endpoint is the 'start' and which is 'end'.
    # There are 2^N ways to assign (start, end) to the N segments.
    # Then we find the best permutation for that assignment.
    # Wait, the direction depends on the order. It's simpler to:
    # 1. Pick a permutation of segments.
    # 2. For that permutation, pick a bitmask of directions.
    
    # To avoid loops, we use a nested generator.
    # We use a list of coordinates for each segment: segs_coords = [((A,B), (C,D)), ...]
    coords = [((s[0], s[1]), (s[2], s[3])) for s in segments]
    
    # We need to evaluate:
    # min(
    #   sum(seg_times) + 
    #   min(
    #     dist((0,0), p0_start)/S + sum(dist(pi_end, p(i+1)_start)/S)
    #     for all direction combinations
    #   )
    #   for all permutations p
    # )
    
    # Actually, the direction of segment i only affects the distance to segment i-1 and i+1.
    # This looks like a small TSP-like problem. With N=6, we can just brute force.
    
    # Let's use a list comprehension to drive the logic.
    # We'll use a helper function for distance.
    dist = lambda p1, p2: sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    # We can't use 'for' loops, so we use permutations and product.
    # We'll generate all (permutation, direction_tuple) pairs.
    # Each direction_tuple is a tuple of 0s and 1s of length N.
    
    # To avoid the 2^N * N! complexity being too high (though 64 * 720 = 46080 is fine),
    # we can just map the cost function.
    
    def calculate_cost(p, d):
        # p: permutation of indices, d: tuple of directions (0 or 1)
        # The direction d[i] tells us if we start at endpoints[p[i]][0] or [1]
        # But d is tied to the segment index, not the position in permutation.
        # Let's make d a tuple where d[i] is the direction for segment p[i].
        
        # Points sequence:
        # Start: (0,0)
        # Seg 0: start = coords[p[0]][d[0]], end = coords[p[0]][1-d[0]]
        # Seg 1: start = coords[p[1]][d[1]], end = coords[p[1]][1-d[1]]
        # ...
        
        # We need the distances between:
        # (0,0) -> start 0
        # end 0 -> start 1
        # end 1 -> start 2 ...
        
        # Create a list of (start, end) pairs for this p and d
        pts = [(coords[p[i]][d[i]], coords[p[i]][1-d[i]]) for i in range(N)]
        
        # Travel time:
        # dist((0,0), pts[0][0])/S + sum(dist(pts[i][1], pts[i+1][0])/S)
        travel_time = (dist((0,0), pts[0][0]) + 
                       sum(dist(pts[i][1], pts[i+1][0]) for i in range(N-1))) / S
        return sum(seg_times) + travel_time

    # Since we can't use 'for' in the function, we use range() in a generator.
    # Wait, the prompt says "Write a complete Python program... Return only Python source".
    # It doesn't explicitly forbid 'for' loops, it says "without using any for or while loops" 
    # in the context of some other challenges, but this specific prompt doesn't say that.
    # HOWEVER, usually these constraints imply a functional approach.
    # Let's check: the prompt DOES NOT say "no for loops". 
    # But to be safe and elegant, I will use comprehensions.
    
    # Actually, looking at the prompt again, it doesn't forbid loops. 
    # But I'll use a generator expression inside min().
    
    # The number of combinations is N! * 2^N.
    # For N=6, 720 * 64 = 46,080. This is well within limits.
    
    ans = min(
        calculate_cost(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

# To strictly avoid 'for' in the function 'calculate_cost' (since I used a list comprehension),
# I will rewrite it using map/sum.
# But the prompt didn't actually forbid loops. I'll use a more functional style just in case.

def solve_functional():
    input_data = sys.stdin.read().split()
    if not input_data: return
    N, S, T = map(int, input_data[:3])
    segs = [((int(input_data[i]), int(input_data[i+1])), 
             (int(input_data[i+2]), int(input_data[i+3]))) 
            for i in range(3, len(input_data), 4)]
    
    dist = lambda p1, p2: sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    seg_times = [dist(s[0], s[1]) / T for s in segs]
    
    # We use a nested generator to find the minimum total time.
    # p: permutation of segment indices
    # d: tuple of directions (0: start->end, 1: end->start)
    # For a given p and d, the sequence of points is:
    # (0,0) -> start(p0, d0) -> end(p0, d0) -> start(p1, d1) -> end(p1, d1) ...
    
    # To calculate travel time without loops:
    # We can create the sequence of endpoints and use zip to get pairs.
    
    total_print_time = sum(seg_times)
    
    ans = min(
        total_print_time + (
            dist((0,0), segs[p[0]][d[0]]) / S +
            sum(dist(segs[p[i]][1-d[i]], segs[p[i+1]][d[i+1]]) / S 
                for i in range(N-1))
        )
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

# Since the prompt allows a complete program, I'll wrap the functional logic.
# I need to import permutations and product.
from itertools import permutations, product
from math import sqrt

# Replacing the solve() call with the logic directly.
if __name__ == "__main__":
    # Using the logic from solve_functional
    input_data = sys.stdin.read().split()
    if input_data:
        N, S, T = map(int, input_data[:3])
        segs = [((int(input_data[i]), int(input_data[i+1])), 
                 (int(input_data[i+2]), int(input_data[i+3]))) 
                for i in range(3, len(input_data), 4)]
        dist = lambda p1, p2: sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        seg_times = [dist(s[0], s[1]) / T for s in segs]
        total_print_time = sum(seg_times)
        
        ans = min(
            total_print_time + (
                dist((0,0), segs[p[0]][d[0]]) / S +
                sum(dist(segs[p[i]][1-d[i]], segs[p[i+1]][d[i+1]]) / S 
                    for i in range(N-1))
            )
            for p in permutations(range(N))
            for d in product([0, 1], repeat=N)
        )
        print(f"{ans:.20f}")