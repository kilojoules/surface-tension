import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        ((int(input_data[3 + i*4]), int(input_data[4 + i*4])), 
         (int(input_data[5 + i*4]), int(input_data[6 + i*4])))
        for i in range(N)
    ]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # We need to try all permutations of segments and all possible directions for each segment.
    # There are N! permutations and 2^N direction combinations.
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of the indices of segments
    perms = permutations(range(N))
    
    # For a specific permutation of segments, we want to find the minimum time.
    # We can use a bitmask to represent the directions (0: p1->p2, 1: p2->p1).
    # However, it's cleaner to just iterate through all 2^N combinations.
    
    # We use a list comprehension to calculate the total time for every permutation and direction set.
    # The structure is: for each permutation in perms, for each direction set in range(2^N).
    
    results = [
        sum(
            # Time to move from current position to the start of the segment (at speed S)
            # plus time to print the segment (at speed T).
            # We use a trick with reduce to keep track of the current laser position.
            # But since we can't use loops, we'll pre-calculate the path.
            0 # Placeholder for the sum logic
        )
        for p in perms
    ]
    
    # To avoid loops and use reduce for the path calculation:
    # For a fixed permutation 'p' and a fixed direction set 'mask':
    # We build a list of (start_point, end_point) for the segments.
    
    def calculate_time(p, mask):
        # Determine the directed segments based on the mask
        directed = [
            (segments[p[i]][0], segments[p[i]][1]) if (mask & (1 << i)) == 0 
            else (segments[p[i]][1], segments[p[i]][0])
            for i in range(N)
        ]
        
        # Use reduce to simulate the movement and accumulate time.
        # State: (current_position, total_time)
        final_state = reduce(
            lambda state, seg: (
                seg[1], 
                state[1] + dist(state[0], seg[0]) / S + dist(seg[0], seg[1]) / T
            ),
            directed,
            ((0, 0), 0.0)
        )
        return final_state[1]

    # Now we evaluate calculate_time for all permutations and all masks.
    # We use a nested list comprehension.
    all_times = [
        calculate_time(p, mask)
        for p in perms
        for mask in range(1 << N)
    ]
    
    # Output the minimum of all calculated times.
    print(f"{min(all_times):.20f}")

if __name__ == "__main__":
    solve()