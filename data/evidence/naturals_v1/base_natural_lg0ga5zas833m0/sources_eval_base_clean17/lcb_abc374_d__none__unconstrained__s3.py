import sys
import itertools
import math

def solve():
    # Read N, S, T
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, s, t = map(int, line1)
    except ValueError:
        return

    segments = []
    for _ in range(n):
        segments.append(list(map(int, sys.stdin.readline().split())))

    # Each segment i has two endpoints: P_{i,0} and P_{i,1}
    # A segment is defined by (A_i, B_i) and (C_i, D_i)
    points = []
    for i in range(n):
        points.append((segments[i][0], segments[i][1]))
        points.append((segments[i][2], segments[i][3]))

    # Precompute distance between all endpoints and the starting point (0,0)
    # Distances are needed for:
    # 1. Moving from current endpoint to the start of the next segment (speed S)
    # 2. Printing the segment itself (speed T)
    
    # segment_len[i] is the time to print the i-th segment
    segment_len_time = []
    for i in range(n):
        dx = segments[i][0] - segments[i][2]
        dy = segments[i][1] - segments[i][3]
        dist = math.sqrt(dx*dx + dy*dy)
        segment_len_time.append(dist / t)

    # We need to visit all N segments. For each segment, we can choose:
    # - Which endpoint to start at
    # - Which order to visit the segments
    
    # Since N is small (N <= 6), we can iterate through all permutations of segments
    # and all combinations of starting endpoints.
    
    min_total_time = float('inf')
    
    # All permutations of indices 0 to N-1
    for p in itertools.permutations(range(n)):
        # For each segment in the permutation, we have 2 choices for the direction
        # There are 2^N combinations of directions
        for directions in itertools.product([0, 1], repeat=n):
            current_time = 0.0
            curr_x, curr_y = 0.0, 0.0
            
            for i in range(n):
                seg_idx = p[i]
                # Determine start and end points based on direction
                if directions[i] == 0:
                    start_x, start_y = segments[seg_idx][0], segments[seg_idx][1]
                    end_x, end_y = segments[seg_idx][2], segments[seg_idx][3]
                else:
                    start_x, start_y = segments[seg_idx][2], segments[seg_idx][3]
                    end_x, end_y = segments[seg_idx][0], segments[seg_idx][1]
                
                # Time to move from current position to start of segment (speed S)
                dist_to_start = math.sqrt((start_x - curr_x)**2 + (start_y - curr_y)**2)
                current_time += dist_to_start / s
                
                # Time to print the segment (speed T)
                current_time += segment_len_time[seg_idx]
                
                # Update current position to the end of the segment
                curr_x, curr_y = end_x, end_y
            
            if current_time < min_total_time:
                min_total_time = current_time
                
    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()