import sys
import math
from itertools import permutations, product

def solve():
    # Read input and parse values
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples [(x1, y1), (x2, y2)]
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    # Helper to calculate Euclidean distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute the time taken to print each segment (length / T)
    print_times = [dist(s[0], s[1]) / T for s in segments]
    total_print_time = sum(print_times)

    # We need to try all permutations of segments and all possible directions (start/end)
    # There are N! permutations and 2^N direction combinations
    # Since N <= 6, N! * 2^N is at most 720 * 64 = 46,080, which is small enough.
    
    # Generate all permutations of indices 0 to N-1
    indices_perms = permutations(range(N))
    # Generate all possible binary choices for direction (0: start->end, 1: end->start)
    directions = product([0, 1], repeat=N)

    # Function to calculate the total travel time (non-printing) for a specific order and direction
    def calculate_travel_time(perm, direct):
        # Get the sequence of points visited
        # For each segment i in perm, if direct[i]==0, we go from s[0] to s[1]
        # If direct[i]==1, we go from s[1] to s[0]
        
        # Create a list of (start_point, end_point) for the chosen permutation and direction
        ordered_segments = [
            (segments[i][0] if direct[i] == 0 else segments[i][1],
             segments[i][1] if direct[i] == 0 else segments[i][0])
            for i in perm
        ]
        
        # The travel path is: (0,0) -> start1 -> end1 -> start2 -> end2 ...
        # We only care about the travel time between segments (and from origin to first start)
        # Travel legs: 
        # 1. (0,0) to ordered_segments[0][0]
        # 2. ordered_segments[i][1] to ordered_segments[i+1][0] for i in 0..N-2
        
        # Using a list comprehension to gather all travel distances
        legs = [dist((0, 0), ordered_segments[0][0])] + [
            dist(ordered_segments[i][1], ordered_segments[i+1][0])
            for i in range(N - 1)
        ]
        
        return sum(legs) / S

    # We want to minimize the travel time across all permutations and direction combinations.
    # Since the print time is constant regardless of order, we just add it at the end.
    # We use a generator expression inside min() to avoid creating a large list in memory.
    
    # Note: directions needs to be re-generated or stored because it's an iterator.
    # We can nest the loops using a generator expression.
    
    min_travel_time = min(
        calculate_travel_time(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )

    print(f"{min_travel_time + total_print_time:.20f}")

if __name__ == "__main__":
    solve()