import sys
from itertools import permutations
import math

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    coords = []
    for i in range(N):
        # Each segment is (x1, y1, x2, y2)
        coords.append(list(map(int, input_data[3 + i*4 : 7 + i*4])))

    # Precompute lengths of segments
    # lengths[i] = distance between (A_i, B_i) and (C_i, D_i)
    lengths = [math.sqrt((c[0]-c[2])**2 + (c[1]-c[3])**2) for c in coords]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to decide the order of segments and the direction of each segment.
    # N is small (up to 6), so we can iterate through all permutations of segments (N!)
    # and all combinations of directions (2^N).
    # Total complexity: N! * 2^N * N, which is 720 * 64 * 6 approx 276,480.
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # For each permutation, we use a bitmask to represent the direction.
    # 0: (A, B) -> (C, D), 1: (C, D) -> (A, B)
    
    # To minimize loops and recursion, we use list comprehensions.
    # We calculate the total time for one specific configuration.
    
    def calculate_time(perm, mask):
        # Current position
        curr_x, curr_y = 0, 0
        total_time = 0.0
        
        for i in range(N):
            idx = perm[i]
            # Determine start and end points based on mask
            # mask >> i & 1 decides direction for the i-th segment in the permutation
            if (mask >> i) & 1:
                start_p = (coords[idx][2], coords[idx][3])
                end_p = (coords[idx][0], coords[idx][1])
            else:
                start_p = (coords[idx][0], coords[idx][1])
                end_p = (coords[idx][2], coords[idx][3])
            
            # Time to move to start point (at speed S) + time to print (at speed T)
            total_time += dist((curr_x, curr_y), start_p) / S
            total_time += lengths[idx] / T
            curr_x, curr_y = end_p
            
        return total_time

    # Use a generator to find the minimum time across all permutations and masks
    # We can't use for/while loops for logic, but we can use min() with a generator
    ans = min(
        calculate_time(p, m) 
        for p in all_perms 
        for m in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()