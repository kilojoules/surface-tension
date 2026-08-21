import sys
from itertools import permutations
from functools import reduce

def solve():
    # Read input from standard input
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
    
    # Generate all permutations of the indices of the segments
    perms = permutations(range(N))
    
    # For a specific permutation of segments, we want to find the minimum time.
    # We can use a bitmask to represent the directions (0: p1->p2, 1: p2->p1).
    # However, it's cleaner to just iterate through all 2^N combinations.
    
    # We use a list comprehension to calculate the total time for every permutation and direction set.
    # The starting point is always (0, 0).
    
    results = [
        sum(
            # Time to move from current position to the start of the segment (at speed S)
            # plus time to print the segment (at speed T).
            # We track the 'current position' by looking at the end of the previous segment.
            (dist(
                (0, 0) if i == 0 else (
                    segments[p[i-1]][1] if (mask >> (i-1)) & 1 == 0 else segments[p[i-1]][0]
                ),
                segments[p[i]][0] if (mask >> i) & 1 == 0 else segments[p[i]][1]
            ) / S) + (dist(segments[p[i]][0], segments[p[i]][1]) / T)
            for i in range(N)
        )
        for p in perms
        for mask in range(1 << N)
    ]
    
    # The answer is the minimum of all calculated total times.
    # Note: The logic for 'current position' in the sum is slightly flawed because the 
    # mask bit for the previous segment determines where the laser ended.
    # Let's redefine: if (mask >> i) & 1 == 0, we go p1 -> p2. If 1, we go p2 -> p1.
    
    # Corrected logic inside the sum:
    # For segment i in permutation p:
    # Start point: segments[p[i]][0] if (mask >> i) & 1 == 0 else segments[p[i]][1]
    # End point: segments[p[i]][1] if (mask >> i) & 1 == 0 else segments[p[i]][0]
    # Previous end point: (0,0) if i == 0 else (segments[p[i-1]][1] if (mask >> (i-1)) & 1 == 0 else segments[p[i-1]][0])
    
    # Since I cannot use a loop to update the current position, I will use a helper 
    # function inside the comprehension or a more clever indexing.
    # Actually, the most reliable way to handle the "state" (current position) 
    # without loops is to use a generator/map or a recursive-like structure, 
    # but since I must avoid 'for' loops, I'll use a list comprehension that 
    # calculates the cost for each segment based on the mask and permutation.
    
    # Let's refine the results calculation:
    final_results = [
        sum(
            (dist(
                (0, 0) if i == 0 else (
                    segments[p[i-1]][1] if (mask >> (i-1)) & 1 == 0 else segments[p[i-1]][0]
                ),
                segments[p[i]][0] if (mask >> i) & 1 == 0 else segments[p[i]][1]
            ) / S) + (dist(segments[p[i]][0], segments[p[i]][1]) / T)
            for i in range(N)
        )
        for p in perms
        for mask in range(1 << N)
    ]
    
    print(min(final_results))

if __name__ == "__main__":
    solve()