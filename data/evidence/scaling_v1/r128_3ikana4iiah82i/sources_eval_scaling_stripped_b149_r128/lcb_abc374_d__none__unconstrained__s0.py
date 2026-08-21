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
    
    # Parse line segments into a list of tuples ((x1, y1), (x2, y2))
    segments = [
        ((int(input_data[3 + i*4]), int(input_data[4 + i*4])), 
         (int(input_data[5 + i*4]), int(input_data[6 + i*4])))
        for i in range(N)
    ]

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # For a given permutation of segments, we need to decide the direction of each segment.
    # There are 2^N possible direction combinations.
    # We can use a list comprehension to evaluate all 2^N combinations for a specific permutation.
    
    # To avoid explicit loops, we use a generator expression inside min()
    # We iterate through all permutations of the segments.
    # For each permutation, we iterate through all binary combinations (0 or 1) representing directions.
    
    # We use a list of indices for the binary combinations to avoid 'for' loops.
    # Since N <= 6, 2^6 = 64, which is small.
    
    # We define a function that calculates the total time for a specific order and direction set.
    def calc_time(perm, directions):
        # Current position starts at (0, 0)
        # We use reduce to simulate the movement across the segments.
        # State: (current_pos, total_time)
        
        # Map directions to actual start/end points for the segments in the permutation
        # if directions[i] == 0: start=p1, end=p2; else: start=p2, end=p1
        oriented = [
            (perm[i][0], perm[i][1]) if directions[i] == 0 else (perm[i][1], perm[i][0])
            for i in range(N)
        ]
        
        def step(state, segment):
            pos, time = state
            start, end = segment
            # Time to move to start + time to print to end
            return (end, time + dist(pos, start)/S + dist(start, end)/T)
        
        final_state = reduce(step, oriented, ((0, 0), 0.0))
        return final_state[1]

    # Generate all possible direction bit-strings as tuples
    all_directions = [tuple((i >> j) & 1 for j in range(N)) for i in range(1 << N)]
    
    # The main logic: 
    # 1. Permute the segments.
    # 2. For each permutation, try all direction combinations.
    # 3. Find the minimum time.
    
    # We use a nested generator expression.
    # The outer layer is permutations, the inner layer is the direction combinations.
    ans = min(
        calc_time(p, d)
        for p in permutations(segments)
        for d in all_directions
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()