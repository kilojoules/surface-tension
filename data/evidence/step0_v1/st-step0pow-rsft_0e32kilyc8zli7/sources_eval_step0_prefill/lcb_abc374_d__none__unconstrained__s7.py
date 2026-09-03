import sys
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
        a = float(input_data[idx])
        b = float(input_data[idx+1])
        c = float(input_data[idx+2])
        d = float(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precompute lengths of segments
    # Time to print segment i is length / T
    seg_times = []
    for seg in segments:
        p1, p2 = seg
        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
        seg_times.append(dist / T)

    # We need to visit all N segments. 
    # For each segment, we can start at either endpoint and end at the other.
    # There are N! permutations of segments and 2^N choices of directions.
    # Since N is small (up to 6), we can iterate through all permutations and directions.
    
    min_total_time = float('inf')
    
    # Generate all permutations of segment indices
    for p in permutations(range(N)):
        # For each permutation, there are 2^N ways to choose the direction of printing
        # We can use a bitmask to represent directions: 0 for p1->p2, 1 for p2->p1
        for mask in range(1 << N):
            current_time = 0.0
            curr_pos = (0.0, 0.0)
            
            for i in range(N):
                seg_idx = p[i]
                p1, p2 = segments[seg_idx]
                
                # Determine start and end points based on the mask
                if (mask >> i) & 1:
                    start_pt, end_pt = p2, p1
                else:
                    start_pt, end_pt = p1, p2
                
                # Time to move from current position to start point (speed S)
                move_dist = ((curr_pos[0] - start_pt[0])**2 + (curr_pos[1] - start_pt[1])**2)**0.5
                current_time += move_dist / S
                
                # Time to print the segment (speed T)
                current_time += seg_times[seg_idx]
                
                # Update current position to the end point
                curr_pos = end_pt
            
            if current_time < min_total_time:
                min_total_time = current_time
                
    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()