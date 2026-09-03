import sys
import math
from itertools import permutations, product

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    # Precompute lengths of segments
    lengths = [math.sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) for s in segments]
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to find the optimal order of segments and the direction of each segment.
    # N is small (up to 6), so we can iterate through all permutations and all 2^N directions.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    # Generate all possible direction choices (0: start->end, 1: end->start)
    directions = product([0, 1], repeat=N)
    
    # To avoid nested loops and recursion, we use a generator expression inside min()
    # For each permutation 'p' and each direction configuration 'd':
    # We calculate the total time.
    
    # We need to track the current position. 
    # Let's define a function to calculate the time for a specific sequence.
    def calculate_time(p, d):
        current_pos = (0, 0)
        total_time = 0.0
        for i in range(N):
            seg_idx = p[i]
            dir_idx = d[i]
            
            p1 = segments[seg_idx][0]
            p2 = segments[seg_idx][1]
            
            # Determine start and end based on direction
            start_node = p1 if dir_idx == 0 else p2
            end_node = p2 if dir_idx == 0 else p1
            
            # Time to move to start_node (at speed S) + time to print to end_node (at speed T)
            total_time += dist(current_pos, start_node) / S + lengths[seg_idx] / T
            current_pos = end_node
            
        return total_time

    # Use a list comprehension to evaluate all possibilities and find the minimum
    # Note: Since we can't use for/while loops, we use map or generator expressions.
    ans = min(
        calculate_time(p, d) 
        for p in permutations(range(N)) 
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()