import sys
from itertools import permutations
import math

def solve():
    # Read input and parse N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples: ((x1, y1), (x2, y2))
    segments = [
        (
            (int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
            (int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        )
        for i in range(N)
    ]

    # Precompute lengths of each segment (time taken to print)
    # length / T
    seg_times = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2) / T
        for s in segments
    ]

    # Function to calculate distance between two points
    # distance / S
    def dist_time(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) / S

    # We need to visit every segment. For each segment, we can start at either end.
    # There are N! permutations of segments and 2^N choices of direction.
    # Since N is small (<= 6), we can iterate through all permutations.
    # For the direction, we can use recursion or a bitmask, but since we 
    # process segments sequentially, we can use a recursive function to 
    # find the best direction for the remaining segments.

    # To avoid redundant calculations in the recursion, we can use a helper
    # However, with N=6, the total states are 6! * 2^6 = 720 * 64 = 46,080.
    # This is small enough to compute directly.

    def get_min_time(perm):
        # For a fixed order of segments, find the best direction for each.
        # state: (index_of_segment, current_position)
        # Since we only have 2 choices per segment, we can use recursion.
        
        memo = {}

        def recurse(idx, current_pos):
            state = (idx, current_pos)
            if state in memo:
                return memo[state]
            
            if idx == N:
                return 0
            
            seg = segments[perm[idx]]
            p1, p2 = seg
            t_print = seg_times[perm[idx]]
            
            # Option 1: Move to p1, then print to p2
            res1 = dist_time(current_pos, p1) + t_print + recurse(idx + 1, p2)
            # Option 2: Move to p2, then print to p1
            res2 = dist_time(current_pos, p2) + t_print + recurse(idx + 1, p1)
            
            ans = min(res1, res2)
            memo[state] = ans
            return ans

        return recurse(0, (0, 0))

    # Try all permutations of segment indices
    all_perms = permutations(range(N))
    # For each permutation, calculate the minimum time required
    # We use a list comprehension to calculate the min time for each permutation
    # and then take the minimum of those results.
    
    # Note: The recurse function inside get_min_time handles the 2^N direction choices.
    # We wrap the logic in a function to avoid loops and maintain the requested structure.
    
    results = [get_min_time(p) for p in all_perms]
    print(f"{min(results):.20f}")

if __name__ == "__main__":
    solve()