import sys
from itertools import permutations
import math

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, S, T
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments
    segments = []
    idx = 3
    for _ in range(N):
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))
        idx += 4

    # Precompute lengths of segments (time taken to print)
    # length / T
    seg_times = [math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / T for p1, p2 in segments]
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to decide the order of segments and the direction of each segment.
    # N is small (up to 6), so we can iterate through all permutations (N!) 
    # and all direction combinations (2^N).
    # Total complexity: O(N! * 2^N * N)
    
    # Generate all permutations of segment indices
    all_orders = list(permutations(range(N)))
    
    # For each order, we use dynamic programming or recursion to find the min time.
    # Since we can't use loops/recursion for the 2^N part easily without 
    # violating "no for/while" if interpreted strictly (though the prompt 
    # allows list comprehensions), we can use a recursive-like structure 
    # via map/reduce or simply iterate since the prompt asks for a program.
    # Wait, the prompt doesn't forbid loops, it just asks for the program.
    
    # To handle the 2^N directions without explicit nested loops for each permutation:
    # For a fixed order, we can use DP.
    # dp[i][0] = min time to finish i-th segment ending at its first endpoint
    # dp[i][1] = min time to finish i-th segment ending at its second endpoint
    
    def calculate_min_time(order):
        # Initial state: from (0,0) to start of first segment
        p0 = (0, 0)
        s0 = segments[order[0]]
        # Option 0: p0 -> s0[1] -> s0[0] (ends at s0[0])
        # Option 1: p0 -> s0[0] -> s0[1] (ends at s0[1])
        dp0 = (dist(p0, s0[1]) / S) + seg_times[order[0]]
        dp1 = (dist(p0, s0[0]) / S) + seg_times[order[0]]
        
        # Transition for the rest of the segments
        def transition(state, next_idx):
            prev_dp0, prev_dp1 = state
            curr_seg = segments[next_idx]
            curr_time = seg_times[next_idx]
            
            # To end at curr_seg[0], we must have come from curr_seg[1]
            # We could have reached curr_seg[1] from prev_seg[0] or prev_seg[1]
            # But we need to know which point was the "end" of the previous segment.
            # Let's redefine: 
            # state is (min_time_ending_at_P_i, min_time_ending_at_Q_i)
            # where P_i, Q_i are endpoints of segment i.
            
            # To end at P_i: move from (prev_end) to Q_i, then print Q_i -> P_i
            # We don't know prev_end, it's encoded in the state.
            # Let's use the indices of the endpoints of the previous segment in the order.
            return (0, 0) # Placeholder

        # Since I can't use loops, I'll use a fold-like approach with a helper function
        # But the prompt allows for loops in a standard Python program.
        # I will use a standard approach.
        
        # Let's use a list comprehension to simulate the DP process by 
        # building the DP table.
        
        # We need the coordinates of the endpoints for the current order
        pts = [segments[i] for i in order]
        
        # dp[i] = (min_time_ending_at_pts[i][0], min_time_ending_at_pts[i][1])
        # We can't use a loop to fill the DP table. We can use a recursive function.
        # However, recursion is allowed.
        
        memo = {}
        def get_min(idx, end_at_first):
            state = (idx, end_at_first)
            if state in memo: return memo[state]
            
            curr_seg = pts[idx]
            curr_len_time = seg_times[order[idx]]
            
            # To end at 'end_at_first', we must start at 'not end_at_first'
            start_pt = curr_seg[0] if end_at_first else curr_seg[1]
            
            if idx == 0:
                res = (dist((0,0), start_pt) / S) + curr_len_time
            else:
                # Previous segment ended at either its 0-th or 1-st endpoint
                prev_seg = pts[idx-1]
                # Option A: prev ended at prev_seg[0]
                # Option B: prev ended at prev_seg[1]
                res = min(
                    get_min(idx-1, True) + (dist(prev_seg[0], start_pt) / S),
                    get_min(idx-1, False) + (dist(prev_seg[1], start_pt) / S)
                ) + curr_len_time
            
            memo[state] = res
            return res

        return min(get_min(N-1, True), get_min(N-1, False))

    # To avoid recursion limits and loops, I'll use a list comprehension to 
    # iterate through permutations and a helper to calculate the path.
    # Since I can use recursion, I'll use it.
    
    # To find the absolute minimum across all permutations:
    results = [calculate_min_time(order) for order in all_orders]
    print(f"{min(results):.20f}")

if __name__ == "__main__":
    solve()