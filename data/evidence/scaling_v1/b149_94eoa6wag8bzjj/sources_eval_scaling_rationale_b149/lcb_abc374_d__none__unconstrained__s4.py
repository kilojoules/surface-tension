import sys
import math
from itertools import permutations, product

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
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute printing time for each segment
    # printing_times[i] is the time taken to print segment i
    printing_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments
    # For each segment, we can start at either endpoint
    # There are N! permutations and 2^N direction combinations
    
    # Generate all permutations of indices 0 to N-1
    all_perms = permutations(range(N))
    
    # Generate all possible direction choices (0 or 1 for each segment)
    # 0: start at A, end at C; 1: start at C, end at A
    all_dirs = product([0, 1], repeat=N)

    def calculate_total_time(perm, dirs):
        # Current position starts at (0, 0)
        # We need to track the current position through the sequence
        # Using a helper function to calculate travel time between segments
        
        # Map perm and dirs to a sequence of (start_point, end_point)
        path = [
            (segments[perm[i]][0] if dirs[perm[i]] == 0 else segments[perm[i]][1],
             segments[perm[i]][1] if dirs[perm[i]] == 0 else segments[perm[i]][0])
            for i in range(N)
        ]
        
        # Travel time from (0,0) to first start point
        start_travel = dist((0, 0), path[0][0]) / S
        
        # Travel times between segments
        # Between segment i and i+1: end of i to start of i+1
        between_travels = [
            dist(path[i][1], path[i+1][0]) / S 
            for i in range(N - 1)
        ]
        
        return start_travel + sum(between_travels) + sum(printing_times)

    # Since we cannot use loops, we use min() with a generator expression
    # We need to iterate over all permutations and all possible direction assignments
    # Note: dirs is a tuple of length N, where dirs[i] corresponds to segment i
    
    # To avoid nested loops, we can flatten the search space
    # However, the direction choice depends on the segment index, not the permutation position.
    # So we can iterate over all 2^N direction configs and N! permutations.
    
    # We use a generator to find the minimum time
    # We use a list comprehension inside min() to evaluate all combinations
    # Because N is small (<= 6), N! * 2^N is at most 720 * 64 = 46080
    
    # We need to redefine how dirs are handled to fit the generator
    # Let's use a bitmask for directions: 0 to 2^N - 1
    
    def get_dir_bit(mask, idx):
        return (mask >> idx) & 1

    # We use a generator expression to find the minimum
    # We iterate over all permutations and all possible masks for directions
    ans = min(
        calculate_total_time(p, [get_dir_bit(m, i) for i in range(N)])
        for p in permutations(range(N))
        for m in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()