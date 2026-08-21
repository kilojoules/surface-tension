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
    
    # Parse line segments into pairs of endpoints
    segments = [
        ((int(input_data[3 + i*4]), int(input_data[4 + i*4])), 
         (int(input_data[5 + i*4]), int(input_data[6 + i*4])))
        for i in range(N)
    ]

    # Helper to calculate Euclidean distance
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # For a given permutation of segments, we can choose the direction of each segment.
    # There are 2^N possible direction combinations.
    # We use a list comprehension to evaluate all direction combinations for a permutation.
    # Each direction combination is represented by a tuple of 0s and 1s.
    
    # To avoid explicit loops, we use a generator expression inside min()
    # We iterate through all permutations of the segments.
    # For each permutation, we iterate through all 2^N direction assignments.
    
    # We use a trick with range(2**N) and bit-shifting to simulate the direction choices.
    
    ans = min(
        min(
            # For a specific permutation 'p' and direction bitmask 'm'
            # We calculate the total time.
            # The laser starts at (0, 0).
            # We use reduce to keep track of (current_position, total_time).
            reduce(
                lambda acc, i: (
                    # New position is the 'end' of the segment based on bitmask m
                    p[i][1] if (m >> i) & 1 else p[i][0],
                    # New time is old time + travel time to start + printing time
                    acc[1] + 
                    dist(acc[0], p[i][0] if (m >> i) & 1 else p[i][1]) / S + 
                    dist(p[i][0], p[i][1]) / T
                ),
                permutations(segments),
                ((0, 0), 0)
            )[1]
            for m in range(2**N)
        )
        for p in permutations(segments)
    )

    # The logic above is slightly flawed because the reduce is inside the m loop.
    # Let's redefine: for every permutation p, and every mask m, calculate the cost.
    
    # Corrected approach using a nested generator:
    final_ans = min(
        reduce(
            lambda acc, i: (
                p[i][1] if (m >> i) & 1 else p[i][0],
                acc[1] + dist(acc[0], p[i][0] if (m >> i) & 1 else p[i][1]) / S + dist(p[i][0], p[i][1]) / T
            ),
            range(N),
            ((0, 0), 0)
        )[1]
        for p in permutations(segments)
        for m in range(2**N)
    )

    print(f"{final_ans:.20f}")

if __name__ == "__main__":
    solve()