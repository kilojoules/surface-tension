import sys
from itertools import permutations
import math

def solve():
    # Read input and parse N, S, T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        (
            (int(input_data[3 + 2*i]), int(input_data[4 + 2*i])),
            (int(input_data[5 + 2*i]), int(input_data[6 + 2*i]))
        )
        for i in range(N)
    ]
    
    # Precompute lengths of each segment
    # length = sqrt((x1-x2)^2 + (y1-y2)^2)
    # Time to print = length / T
    seg_lengths = [
        math.sqrt((s[0][0] - s[1][0])**2 + (s[0][1] - s[1][1])**2)
        for s in segments
    ]
    
    # Function to calculate distance between two points
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # We need to visit every segment. For each segment i, we can start at 
    # endpoint A_i and end at B_i, or vice versa.
    # There are N! permutations of segments and 2^N ways to choose directions.
    # Since N is small (<= 6), we can iterate through all permutations and 
    # use recursion/bitmask or simply iterate through all direction combinations.
    
    # To handle the 2^N directions without nested loops, we can use a 
    # recursive function or a comprehension. 
    # For a fixed permutation P of indices 0...N-1:
    # Let directions be a tuple of 0 or 1. 
    # If dir[i] == 0: start at segments[P[i]][0], end at segments[P[i]][s1]
    # If dir[i] == 1: start at segments[P[i]][1], end at segments[P[i]][0]
    
    # We use a helper to calculate total time for a specific order and direction set
    def calculate_time(p, dirs):
        # Current position starts at (0, 0)
        curr_pos = (0, 0)
        total_time = 0.0
        
        for i in range(N):
            idx = p[i]
            p1, p2 = segments[idx]
            # Determine start and end points based on direction bit
            start_pt = p1 if dirs[i] == 0 else p2
            end_pt = p2 if dirs[i] == 0 else p1
            
            # Move to start point (speed S)
            total_time += dist(curr_pos, start_pt) / S
            # Print segment (speed T)
            total_time += seg_lengths[idx] / T
            # Update current position
            curr_pos = end_pt
            
        return total_time

    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    
    # For each permutation, we want to find the best direction configuration.
    # Since N is small, we can use a list comprehension to test all 2^N directions.
    # We use a range(1 << N) to represent the 2^N combinations of directions.
    
    # To optimize, we can pre-calculate the distance matrices, but with N=6, 
    # the brute force approach is: 6! * 2^6 = 720 * 64 = 46,080 iterations.
    # This is well within the time limit for Python.
    
    # We use a generator expression inside min() to find the global minimum.
    ans = min(
        calculate_time(p, [(d >> i) & 1 for i in range(N)])
        for p in all_perms
        for d in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()