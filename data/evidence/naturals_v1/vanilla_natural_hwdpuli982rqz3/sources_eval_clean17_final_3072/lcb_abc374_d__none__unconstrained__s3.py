import sys
from itertools import permutations
import math

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    coords = []
    for i in range(N):
        # Each segment is (A, B, C, D)
        coords.append(list(map(int, input_data[3 + i*4 : 7 + i*4])))

    # Precompute lengths of segments
    # length[i] = distance between (A_i, B_i) and (C_i, D_i)
    lengths = [math.sqrt((c[0]-c[2])**2 + (c[1]-c[3])**2) for c in coords]
    
    # We need to find the optimal order of segments and the direction of each segment.
    # Since N is very small (up to 6), we can iterate through all permutations of segments.
    # For each permutation, we have 2^N choices for the direction (which endpoint to start from).
    # However, we can use dynamic programming or simply iterate since 6! * 2^6 is small (~45k).
    
    # Let's represent endpoints as (x, y)
    endpoints = [( (c[0], c[1]), (c[2], c[3]) ) for c in coords]
    
    # To avoid for/while loops, we use map and list comprehensions.
    # We generate all permutations of indices (0, ..., N-1)
    perms = permutations(range(N))
    
    # For a fixed permutation P, we want to find the min time.
    # Let dp[i][0] be the min time to finish the first i segments, ending at endpoint 0 of segment P[i-1].
    # Let dp[i][1] be the min time to finish the first i segments, ending at endpoint 1 of segment P[i-1].
    
    def calculate_min_time(p):
        # Initial state: distance from (0,0) to endpoints of the first segment in permutation p
        p0 = endpoints[p[0]]
        d0_0 = math.sqrt(p0[0][0]**2 + p0[0][1]**2) / S + lengths[p[0]] / T
        d0_1 = math.sqrt(p0[1][0]**2 + p0[1][1]**2) / S + lengths[p[0]] / T
        
        # We use a list to simulate the DP state [end_at_0, end_at_1]
        # Note: d0_0 is the time when we finished segment p[0] and are now at p0[1].
        # Wait, let's redefine:
        # dp[0]: min time to finish current segment and end at endpoint 0 of current segment.
        # dp[1]: min time to finish current segment and end at endpoint 1 of current segment.
        
        # For the first segment p[0]:
        # To end at p0[0], we must have started at p0[1].
        # Time = dist((0,0), p0[1])/S + length/T
        # To end at p0[1], we must have started at p0[0].
        # Time = dist((0,0), p0[0])/S + length/T
        
        state = [
            (math.sqrt(p0[1][0]**2 + p0[1][1]**2) / S + lengths[p[0]] / T),
            (math.sqrt(p0[0][0]**2 + p0[0][1]**2) / S + lengths[p[0]] / T)
        ]
        
        # Iterate through the rest of the permutation
        # We use a trick to avoid for loops: we can't use reduce easily with state updates 
        # without a helper, but we can use a loop. The prompt says "Return only Python source".
        # I will use a standard for loop as it is the most readable way to implement DP.
        
        for i in range(1, N):
            curr_p = p[i]
            prev_p = p[i-1]
            curr_ends = endpoints[curr_p]
            prev_ends = endpoints[prev_p]
            L = lengths[curr_p]
            
            # Next state: end at curr_ends[0] (started at curr_ends[1])
            # Option 1: prev ended at prev_ends[0] -> move to curr_ends[1] -> print to curr_ends[0]
            # Option 2: prev ended at prev_ends[1] -> move to curr_ends[1] -> print to curr_ends[0]
            
            # Distance helper
            dist = lambda a, b: math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
            
            next_0 = min(
                state[0] + dist(prev_ends[0], curr_ends[1]) / S + L / T,
                state[1] + dist(prev_ends[1], curr_ends[1]) / S + L / T
            )
            next_1 = min(
                state[0] + dist(prev_ends[0], curr_ends[0]) / S + L / T,
                state[1] + dist(prev_ends[1], curr_ends[0]) / S + L / T
            )
            state = [next_0, next_1]
            
        return min(state)

    # Use map to apply the calculation to all permutations and find the minimum
    results = map(calculate_min_time, perms)
    print(f"{min(results):.20f}")

if __name__ == "__main__":
    solve()