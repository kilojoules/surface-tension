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
    
    # Parse line segments into a list of tuples: ((x1, y1), (x2, y2))
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
    all_perms = permutations(range(N))
    
    # For a specific permutation of segments, we want to find the minimum time.
    # We can use a bitmask to represent the directions (0: p1->p2, 1: p2->p1).
    # However, it's cleaner to just iterate through all 2^N combinations.
    
    # We use a list comprehension to evaluate all permutations and all direction combinations.
    # For each combination, we calculate the total time.
    
    # To avoid explicit loops, we use a nested comprehension.
    # The outer layer is the permutation of segments.
    # The inner layer is the combination of directions (represented by a tuple of 0s and 1s).
    
    # We pre-calculate the length of each segment to avoid repeated work.
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    
    # The total time is the sum of (distance to start of segment / S) + (length of segment / T).
    # The starting point of the first segment is relative to (0, 0).
    # The starting point of segment i is relative to the end point of segment i-1.
    
    # We use a helper function to calculate the cost of a specific sequence of directed segments.
    def calc_cost(seq):
        # seq is a list of (start_point, end_point)
        # Initial move from (0,0) to seq[0][0]
        initial_move = dist((0, 0), seq[0][0]) / S
        # Moves between segments: from seq[i][1] to seq[i+1][0]
        between_moves = sum(dist(seq[i][1], seq[i+1][0]) / S for i in range(N - 1))
        # Printing time: length of each segment / T
        printing_time = sum(seg_lengths[i] / T for i in range(N))
        return initial_move + between_moves + printing_time

    # Generate all possible directed sequences
    # For each permutation p, there are 2^N ways to orient the segments.
    # We use a product-like approach via list comprehension.
    
    # Since we can't use itertools.product, we can use a range(2**N) and bit-shifting.
    # But we can just use a recursive-like structure or a clever comprehension.
    # Actually, the simplest way to get 2^N is to use a list comprehension over range(2**N).
    
    # Let's define the logic to get the points based on the bitmask.
    # For permutation p and mask m:
    # Segment i in permutation is segments[p[i]].
    # If (m >> i) & 1 == 0: start = segments[p[i]][0], end = segments[p[i]][1]
    # Else: start = segments[p[i]][1], end = segments[p[i]][0]
    
    ans = min(
        calc_cost([
            (segments[p[i]][0] if not (m & (1 << i)) else segments[p[i]][1],
             segments[p[i]][1] if not (m & (1 << i)) else segments[p[i]][0])
            for i in range(N)
        ])
        for p in all_perms
        for m in range(1 << N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()