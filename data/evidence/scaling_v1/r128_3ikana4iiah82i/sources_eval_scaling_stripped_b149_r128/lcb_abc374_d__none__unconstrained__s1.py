import sys
from itertools import permutations
from functools import reduce

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
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Precompute the time taken to print each segment
    # print_times[i] is the time to move from one end to the other at speed T
    print_times = [dist(s[0], s[1]) / T for s in segments]

    # We need to try all permutations of segments and all possible directions for each
    # There are N! permutations and 2^N direction combinations.
    # For N=6, 720 * 64 = 46,080, which is well within limits.
    
    # Generate all permutations of segment indices
    perms = permutations(range(N))
    
    # For a fixed permutation, we want to find the minimum time.
    # We can use a bitmask to represent directions: 0 for (start, end), 1 for (end, start)
    # However, it's easier to just iterate through all 2^N combinations.
    
    # To avoid explicit loops, we use a generator expression inside min()
    # We evaluate every permutation and every possible direction assignment.
    
    ans = min(
        sum(
            # Time to print the current segment
            print_times[p[i]] + 
            # Time to move from the end of the previous segment to the start of the current
            # The previous segment is p[i-1]. 
            # The direction of the current segment is determined by the bitmask 'mask'
            (dist(
                # End point of previous segment
                (segments[p[i-1]][1] if (mask & (1 << (i-1))) == 0 else segments[p[i-1]][0]),
                # Start point of current segment
                (segments[p[i]][0] if (mask & (1 << i)) == 0 else segments[p[i]][1])
            ) / S if i > 0 else 
            # For the first segment, move from (0,0) to the start point
            dist((0, 0), (segments[p[i]][0] if (mask & (1 << i)) == 0 else segments[p[i]][1])) / S
        )
        for p in perms
        for mask in range(1 << N)
        for i in range(N)
    )
    
    # Wait, the logic above is slightly flawed because the 'sum' is inside the 'min'
    # but the 'for i in range(N)' is outside the 'sum'. 
    # Let's rewrite it to correctly sum the costs for a specific permutation and mask.
    
    # Corrected version:
    ans = min(
        sum(
            print_times[p[i]] + (
                dist(
                    (segments[p[i-1]][1] if (mask & (1 << (i-1))) == 0 else segments[p[i-1]][0]),
                    (segments[p[i]][0] if (mask & (1 << i)) == 0 else segments[p[i]][1])
                ) / S if i > 0 else 
                dist((0, 0), (segments[p[i]][0] if (mask & (1 << i)) == 0 else segments[p[i]][1])) / S
            )
            for i in range(N)
        )
        for p in perms
        for mask in range(1 << N)
    )

    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()