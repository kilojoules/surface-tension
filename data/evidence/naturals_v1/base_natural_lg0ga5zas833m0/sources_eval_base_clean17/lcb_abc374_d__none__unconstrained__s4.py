import sys
import math
from itertools import permutations

def solve():
    # Read N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        a = int(input_data[idx])
        b = int(input_data[idx+1])
        c = int(input_data[idx+2])
        d = int(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precalculate length of each segment
    seg_lengths = []
    for p1, p2 in segments:
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        seg_lengths.append(dist)

    # Total time spent printing is constant regardless of order
    print_time = sum(seg_lengths) / T
    
    # We need to find the minimum travel time between segments.
    # Since N is small (up to 6), we can iterate through all permutations of segments
    # and for each segment, try both possible directions (start point, end point).
    
    min_travel_dist = float('inf')
    
    # Permutations of indices 0 to N-1
    for p in permutations(range(N)):
        # There are 2^N ways to choose the direction of each segment
        # We can use recursion or bitmask to iterate through directions
        
        # To optimize, we can use a simple recursive function to try both directions
        # for each segment in the current permutation.
        
        memo_directions = [0] * N
        
        def find_min_dist(step, current_pos):
            nonlocal min_travel_dist
            if step == N:
                # Calculate travel distance for this path
                # Note: we only need the travel distance, print_time is added at the end
                return 0
            
            # This is not quite right for a recursive approach because we need the 
            # total distance. Let's use a different approach.
            return 0

        # Instead of recursion, let's just iterate through 2^N directions manually
        # for the specific permutation p.
        # But with N=6, 6! * 2^6 = 720 * 64 = 46080, which is very small.
        
        # Let's use a simple loop for directions
        for i in range(1 << N):
            travel_dist = 0.0
            curr_x, curr_y = 0.0, 0.0
            
            for step in range(N):
                seg_idx = p[step]
                p1, p2 = segments[seg_idx]
                
                # Determine which endpoint to start from based on bit i
                if (i >> step) & 1:
                    start_node = p1
                    end_node = p2
                else:
                    start_node = p2
                    end_node = p1
                
                # Distance from current position to the start of the segment
                dist_to_start = math.sqrt((curr_x - start_node[0])**2 + (curr_y - start_node[1])**2)
                travel_dist += dist_to_start
                
                # Update current position to the end of the segment
                curr_x, curr_y = end_node[0], end_node[1]
            
            if travel_dist < min_travel_dist:
                min_travel_dist = travel_dist

    # Total time = (Total Travel Distance / S) + (Total Printing Distance / T)
    total_time = (min_travel_dist / S) + print_time
    print(f"{total_time:.20f}")

if __name__ == "__main__":
    solve()