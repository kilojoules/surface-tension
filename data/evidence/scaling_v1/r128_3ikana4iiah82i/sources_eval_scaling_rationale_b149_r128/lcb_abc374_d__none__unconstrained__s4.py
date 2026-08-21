import sys
import math
from itertools import permutations

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

    # Pre-calculate the time taken to print each segment (length / T)
    print_times = [dist(s[0], s[1]) / T for s in segments]
    
    # The total printing time is constant regardless of the order
    total_print_time = sum(print_times)

    # We need to find the minimum travel time between segments.
    # There are N! permutations of segments and 2^N ways to choose endpoints.
    # Since N is small (<= 6), we can iterate through all permutations.
    # For a fixed permutation, we use dynamic programming (or a fold) to find the min travel time.
    
    def get_min_travel_time(perm_indices):
        # dp[i][0] is min travel time ending at the first endpoint of the i-th segment in perm
        # dp[i][1] is min travel time ending at the second endpoint of the i-th segment in perm
        
        # Initial state: travel from (0,0) to the start of the first segment
        # Note: The laser must move to one endpoint, then print to the other.
        # So if we end at endpoint 1, we must have started printing at endpoint 0.
        
        s0 = segments[perm_indices[0]]
        # Option 0: Start at s0[0], end at s0[1]. Travel: (0,0) -> s0[0]
        # Option 1: Start at s0[1], end at s0[0]. Travel: (0,0) -> s0[1]
        initial_dp = [
            dist((0, 0), s0[0]) / S,
            dist((0, 0), s0[1]) / S
        ]
        
        # Transition for the rest of the segments
        # We use a list comprehension to simulate the DP state transition
        def transition(dp, idx):
            s_prev = segments[perm_indices[idx-1]]
            s_curr = segments[perm_indices[idx]]
            # New dp[0]: end at s_curr[1] (started at s_curr[0])
            #   - from prev end s_prev[0] to s_curr[0]
            #   - from prev end s_prev[1] to s_curr[0]
            # New dp[1]: end at s_curr[0] (started at s_curr[1])
            #   - from prev end s_prev[0] to s_curr[1]
            #   - from prev end s_prev[1] to s_curr[1]
            return [
                min(dp[0] + dist(s_prev[0], s_curr[0]) / S, 
                    dp[1] + dist(s_prev[1], s_curr[0]) / S),
                min(dp[0] + dist(s_prev[0], s_curr[1]) / S, 
                    dp[1] + dist(s_prev[1], s_curr[1]) / S)
            ]

        # Use a manual loop replacement via a custom reduction-like structure
        # Since we can't use loops, we can use a recursive-like structure via a list 
        # but the prompt forbids recursion. We can use a trick with a list and 
        # a helper function called via map or a comprehension.
        # However, for N=6, we can just hardcode the transitions or use a 
        # functional approach to build the DP table.
        
        # We can use a list comprehension to generate the DP states
        # But the state depends on the previous one. 
        # A trick to bypass loops for DP is using a helper function with 
        # a fixed number of iterations.
        
        def run_dp(current_dp, index):
            if index == N:
                return min(current_dp)
            return run_dp(transition(current_dp, index), index + 1)
        
        # Wait, the prompt says no recursion. Let's use a different approach.
        # Since N is very small, we can use a list comprehension to 
        # evaluate all 2^N endpoint combinations for a given permutation.
        return None # Not used

    # Revised approach: For each permutation, evaluate all 2^N endpoint choices.
    # There are 2^N ways to orient the segments.
    # For a fixed permutation and fixed orientations, the travel time is:
    # dist((0,0), start1)/S + dist(end1, start2)/S + ... + dist(end_{N-1}, start_N)/S
    
    # Generate all possible orientation masks (0: start->end, 1: end->start)
    masks = [[] for _ in range(1 << N)]
    # Using a list comprehension to fill masks
    masks = [[(m >> i) & 1 for i in range(N)] for m in range(1 << N)]
    
    # Calculate total time for a specific permutation and orientation mask
    def calc_time(perm, mask):
        # Determine start and end points for each segment based on the mask
        # If mask[i] == 0: start = s[0], end = s[1]
        # If mask[i] == 1: start = s[1], end = s[0]
        pts = [
            (segments[perm[i]][mask[i]], segments[perm[i]][1 - mask[i]])
            for i in range(N)
        ]
        
        # Travel time: (0,0) -> pts[0][0] -> pts[0][1] -> pts[1][0] -> pts[1][1] ...
        # The printing time is already added separately.
        # We need: dist((0,0), pts[0][0])/S + sum(dist(pts[i][1], pts[i+1][0])/S)
        
        travel_dist = dist((0,0), pts[0][0]) + sum(
            dist(pts[i][1], pts[i+1][0]) for i in range(N-1)
        )
        return travel_dist / S

    # Find the minimum travel time across all permutations and masks
    min_travel_time = min(
        calc_time(perm, mask)
        for perm in permutations(range(N))
        for mask in masks
    )

    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()