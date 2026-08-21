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
    
    # Parse line segments into pairs of endpoints
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate the time to print each segment (length / T)
    print_times = [dist(s[0], s[1]) / T for s in segments]
    total_print_time = sum(print_times)

    # We need to find the minimum travel time between segments.
    # For each segment, we can start at endpoint 0 and end at 1, or vice versa.
    # There are N! permutations of segments and 2^N choices of directions.
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # For a fixed permutation, we use DP or a similar approach to find the best directions.
    # However, since N is very small (<= 6), we can iterate through all 2^N direction combinations.
    # A direction bit i=0 means segment i is printed from endpoint 0 to 1.
    # A direction bit i=1 means segment i is printed from endpoint 1 to 0.
    
    def get_travel_time(perm, directions):
        # Current position starts at (0, 0)
        curr_pos = (0, 0)
        travel_time = 0
        
        for idx in perm:
            p0, p1 = segments[idx]
            # Determine start and end points based on direction bit
            start_p, end_p = (p0, p1) if (directions >> idx) & 1 == 0 else (p1, p0)
            
            # Time to move from current position to the start of the segment
            travel_time += dist(curr_pos, start_p) / S
            # Update current position to the end of the segment
            curr_pos = end_p
            
        return travel_time

    # We use a generator expression inside min() to find the absolute minimum travel time.
    # We iterate over all permutations and all 2^N direction combinations.
    min_travel_time = min(
        get_travel_time(p, d)
        for p in all_perms
        for d in range(1 << N)
    )

    # Final answer is the sum of the constant printing time and the minimum travel time.
    print(f"{total_print_time + min_travel_time:.20f}")

if __name__ == "__main__":
    solve()