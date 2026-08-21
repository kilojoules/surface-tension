import sys
import itertools
import math

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
    for i in range(N):
        a = float(input_data[idx])
        b = float(input_data[idx+1])
        c = float(input_data[idx+2])
        d = float(input_data[idx+3])
        segments.append(((a, b), (c, d)))
        idx += 4

    # Precompute the time to draw each segment
    draw_times = []
    for seg in segments:
        p1, p2 = seg
        dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        draw_times.append(dist / T)

    # The state can be represented by (mask of printed segments, last_position)
    # Since N is small (up to 6), we can iterate through all permutations of segments
    # and for each segment, decide which endpoint to start from.
    
    min_total_time = float('inf')
    
    # All permutations of segments to print
    for perm in itertools.permutations(range(N)):
        # For each segment in the permutation, there are 2 choices for the starting endpoint
        # 2^N combinations of directions
        for directions in itertools.product([0, 1], repeat=N):
            current_time = 0.0
            curr_pos = (0.0, 0.0)
            
            for i in range(N):
                seg_idx = perm[i]
                p1, p2 = segments[seg_idx]
                
                # Determine start and end points based on direction
                if directions[i] == 0:
                    start_node = p1
                    end_node = p2
                else:
                    start_node = p2
                    end_node = p1
                
                # Time to move to start_node without emitting laser
                move_dist = math.sqrt((curr_pos[0] - start_node[0])**2 + (curr_pos[1] - start_node[1])**2)
                current_time += move_dist / S
                
                # Time to draw the segment
                current_time += draw_times[seg_idx]
                
                # Update current position to the end of the segment
                curr_pos = end_node
            
            if current_time < min_total_time:
                min_total_time = current_time
                
    print(f"{min_total_time:.20f}")

if __name__ == "__main__":
    solve()