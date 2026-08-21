import sys
from itertools import permutations
from math import sqrt

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Parse N, S, T
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments
    segments = []
    idx = 3
    for _ in range(N):
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])), 
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))
        idx += 4

    # Precompute lengths of segments
    lengths = [sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) for p1, p2 in segments]
    
    # Function to calculate distance between two points
    dist = lambda p1, p2: sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # We need to decide the order of segments (N!) and the direction of each segment (2^N)
    # Since N is small (up to 6), we can iterate through all permutations and bitmasks.
    
    # Generate all permutations of segment indices
    perms = list(permutations(range(N)))
    
    # For each permutation, we use a bitmask to decide the direction:
    # 0: p1 -> p2, 1: p2 -> p1
    
    # To find the minimum time, we can use a nested comprehension or map.
    # We calculate the total time for one specific configuration:
    def calculate_time(perm, mask):
        current_pos = (0, 0)
        total_time = 0.0
        
        for i in range(N):
            seg_idx = perm[i]
            p1, p2 = segments[seg_idx]
            length = lengths[seg_idx]
            
            # Determine start and end based on mask
            # mask >> i & 1 decides the direction for the i-th segment in the permutation
            if (mask >> i) & 1:
                start, end = p2, p1
            else:
                start, end = p1, p2
            
            # Time to move to start (at speed S) + time to print (at speed T)
            total_time += dist(current_pos, start) / S + length / T
            current_pos = end
            
        return total_time

    # Use a generator to avoid creating a massive list in memory, 
    # though with N=6, 6! * 2^6 = 720 * 64 = 46,080 which is small.
    ans = min(
        calculate_time(p, m) 
        for p in perms 
        for m in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()