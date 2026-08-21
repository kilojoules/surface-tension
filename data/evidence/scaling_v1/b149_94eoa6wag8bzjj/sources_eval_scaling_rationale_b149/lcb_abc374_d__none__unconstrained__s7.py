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
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute printing time for each segment
    # print_times[i] is the time taken to print segment i
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments
    # For each segment, we can start at either endpoint
    # There are N! permutations and 2^N direction combinations
    
    # Generate all permutations of indices
    all_perms = permutations(range(N))
    
    # For a given permutation and a choice of directions (0 or 1 for each segment)
    # calculate the total time.
    # directions: 0 means start at s[0] end at s[1], 1 means start at s[1] end at s[0]
    
    def calculate_time(perm, dirs):
        # Current position starts at (0, 0)
        # We use a list comprehension to build the sequence of (start, end) points
        points = [
            (segments[perm[i]][dirs[i]], segments[perm[i]][1 - dirs[i]])
            for i in range(N)
        ]
        
        # The total time is the sum of:
        # 1. Time to print all segments (constant regardless of order/direction)
        # 2. Time to move from (0,0) to the first start point
        # 3. Time to move from the end of segment i to the start of segment i+1
        
        # Move times:
        # Start move
        start_move = dist((0, 0), points[0][0]) / S
        
        # Intermediate moves
        # Use a generator expression to sum distances between end of i and start of i+1
        inter_moves = sum(
            dist(points[i][1], points[i+1][0]) / S 
            for i in range(N - 1)
        )
        
        return sum(print_times) + start_move + inter_moves

    # To avoid explicit loops, we use min() over a generator expression
    # that iterates through all permutations and all 2^N direction combinations.
    
    # Since N is small (<= 6), 6! * 2^6 = 720 * 64 = 46080, which is well within limits.
    
    ans = min(
        calculate_time(p, d)
        for p in all_perms
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()