import sys
import math
from itertools import permutations

def solve():
    # Read N, S, T from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        a = float(input_data[idx])
        b = float(input_data[idx+1])
        c = float(input_data[idx+2])
        d = float(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precalculate length of each segment for printing time
    seg_lengths = []
    for p1, p2 in segments:
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        seg_lengths.append(dist)

    # We need to visit each segment exactly once.
    # Each segment has two endpoints. We can start at either and end at the other.
    # Since N is small (N <= 6), we can iterate through all permutations of segments
    # and all possible orientations (start/end endpoint) for each segment.
    
    min_total_time = float('inf')
    
    # Permutations of segment indices
    for p in permutations(range(N)):
        # There are 2^N ways to choose the direction for each segment
        for i in range(1 << N):
            current_time = 0.0
            curr_pos = (0.0, 0.0)
            
            for j in range(N):
                seg_idx = p[j]
                p1, p2 = segments[seg_idx]
                
                # Determine start and end point of the segment based on the bitmask i
                if (i >> j) & 1:
                    start_pt, end_pt = p1, p2
                else:
                    start_pt, end_pt = p2, p1
                
                # Time to move to the start point (no laser)
                dist_to_start = math.sqrt((curr_pos[0]-start_pt[0])**2 + (curr_pos[1]-start_pt[1])**2)
                current_time += dist_to_start / S
                
                # Time to print the segment (laser on)
                current_time += seg_lengths[seg_idx] / T
                
                # Update current position to the end point
                curr_pos = end_pt
            
            if current_time < min_total_time:
                min_total_time = current_time
                
    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()