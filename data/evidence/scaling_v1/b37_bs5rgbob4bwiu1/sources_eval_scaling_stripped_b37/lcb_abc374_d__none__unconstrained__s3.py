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
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    # Helper to calculate Euclidean distance
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of all segments (time taken while emitting laser)
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    
    # We need to visit every segment. For each segment, we can start at either end.
    # There are N! orderings of segments and 2^N choices of direction.
    # Since N is small (<= 6), we can iterate through all permutations and direction combinations.
    
    # Generate all permutations of indices 0...N-1
    indices_perms = permutations(range(N))
    
    # For a fixed order of segments, we want to minimize travel time.
    # Let dp[i][side] be the min time to finish segment i, ending at side (0 or 1).
    # However, with N=6, we can just use a recursive function with memoization 
    # or simply iterate through all 2^N direction combinations for each permutation.
    
    # To avoid 2^N inside the loop, we can use a simple recursion/comprehension 
    # to calculate the best path for a specific permutation.
    
    def get_min_time(perm):
        # state: (current_position, total_time)
        # We start at (0,0). 
        # For the first segment in the permutation, we can start at either end.
        
        # Initial possibilities after printing the first segment:
        # 1. Move (0,0) -> p1, print p1 -> p2
        # 2. Move (0,0) -> p2, print p2 -> p1
        s_idx = perm[0]
        p1, p2 = segments[s_idx]
        # (end_point, time_taken)
        possibilities = [
            (p2, (dist((0, 0), p1) / S) + (seg_lengths[s_idx] / T)),
            (p1, (dist((0, 0), p2) / S) + (seg_lengths[s_idx] / T))
        ]
        
        # For the remaining segments in the permutation
        for i in range(1, N):
            s_idx = perm[i]
            p1, p2 = segments[s_idx]
            length_time = seg_lengths[s_idx] / T
            
            # For each current possibility, we can print the next segment in two ways:
            # Current Pos -> p1, then print p1 -> p2
            # Current Pos -> p2, then print p2 -> p1
            new_possibilities = [
                (p2, t + (dist(pos, p1) / S) + length_time),
                (p1, t + (dist(pos, p2) / S) + length_time)
                for pos, t in possibilities
            ]
            
            # To keep the list size manageable (though with N=6 it's only 2^6), 
            # we only really care about the minimum time to reach p1 and p2.
            # Since we have two possible end points for the current segment, 
            # we can reduce the state to: (min_time_ending_at_p1, min_time_ending_at_p2)
            
            # Actually, the comprehension above creates 2 * len(possibilities) entries.
            # We can group them by the endpoint and keep the minimum.
            # But with N=6, 2^6 = 64, which is tiny. Let's just use a list comprehension.
            possibilities = new_possibilities

        return min(t for pos, t in possibilities)

    # Map the get_min_time function over all permutations and find the global minimum
    ans = min(get_min_time(p) for p in indices_perms)
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()