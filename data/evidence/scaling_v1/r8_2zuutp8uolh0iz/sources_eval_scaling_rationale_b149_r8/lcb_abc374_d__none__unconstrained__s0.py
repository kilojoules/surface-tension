import sys
import math
from itertools import product, permutations

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((int(input_data[idx]), int(input_data[idx+1]), 
                         int(input_data[idx+2]), int(input_data[idx+3])))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of segments and time to print them
    # seg_info = [(point_a, point_b, print_time), ...]
    seg_info = [((s[0], s[1]), (s[2], s[3]), dist((s[0], s[1]), (s[2], s[3])) / T) 
                for s in segments]

    # We need to try all permutations of segments
    # For each segment, we can start at either endpoint A or B.
    # There are N! permutations and 2^N start/end configurations.
    
    # Generate all permutations of indices 0 to N-1
    all_perms = permutations(range(N))
    
    # For a fixed permutation, we want to find the min time.
    # Since N is small (<= 6), we can iterate through all 2^N direction choices.
    # directions: 0 means A -> B, 1 means B -> A
    all_dirs = product([0, 1], repeat=N)
    
    def calculate_time(perm, dirs):
        # Starting point
        curr_pos = (0, 0)
        total_time = 0.0
        
        # Use a list comprehension to calculate transitions and sum them
        # We need to track the current position, so we use a helper 
        # that computes the sequence of segments based on the permutation and directions.
        
        # Ordered segments based on permutation and direction
        ordered = [
            (seg_info[perm[i]][0] if dirs[i] == 0 else seg_info[perm[i]][1],
             seg_info[perm[i]][1] if dirs[i] == 0 else seg_info[perm[i]][0],
             seg_info[perm[i]][2])
            for i in range(N)
        ]
        
        # To avoid loops, we calculate the travel time between segments.
        # Travel 0: (0,0) to start of 1st segment
        # Travel i: end of segment i to start of segment i+1
        # Printing: sum of all print_times
        
        print_time_sum = sum(s[2] for s in ordered)
        
        # Travel times
        # Start to first
        t0 = dist(curr_pos, ordered[0][0]) / S
        
        # Between segments
        # Use a list comprehension to get distances between end of i and start of i+1
        travel_times = [dist(ordered[i][1], ordered[i+1][0]) / S 
                        for i in range(N-1)]
        
        return print_time_sum + t0 + sum(travel_times)

    # We need to minimize over all permutations and all direction combinations.
    # Using nested list comprehensions to find the minimum.
    # Since we can't use loops, we map the calculate_time function.
    
    # To optimize, we can pre-calculate the 2^N direction costs for each permutation.
    # But with N=6, N! * 2^N = 720 * 64 = 46080, which is small enough.
    
    # We use a generator expression inside min() to find the global minimum.
    # Note: all_dirs needs to be recreated or converted to a list because it's an iterator.
    dirs_list = list(product([0, 1], repeat=N))
    
    ans = min(
        calculate_time(p, d) 
        for p in permutations(range(N)) 
        for d in dirs_list
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()