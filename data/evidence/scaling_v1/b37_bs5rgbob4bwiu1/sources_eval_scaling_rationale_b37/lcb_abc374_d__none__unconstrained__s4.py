import sys
import math
from itertools import permutations, product

def solve():
    # Read input and parse values
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into a list of pairs of coordinates
    # segments = [((A1, B1), (C1, D1)), ...]
    segments = [
        ((int(input_data[3 + 2*i]), int(input_data[4 + 2*i])), 
         (int(input_data[5 + 2*i]), int(input_data[6 + 2*i])))
        for i in range(N)
    ]

    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of segments to avoid redundant calculations
    seg_lengths = [dist(s[0], s[1]) for s in segments]

    # We need to try every permutation of segments
    # For each segment, we can start at either endpoint A or B
    # There are N! permutations and 2^N direction combinations
    
    # Generate all permutations of indices 0...N-1
    all_perms = permutations(range(N))
    
    # Generate all possible direction choices (0: A->C, 1: C->A)
    directions = product([0, 1], repeat=N)
    
    # Since we need to iterate over permutations and directions, 
    # and we cannot use loops, we use a nested comprehension 
    # to calculate the total time for every possibility and find the minimum.
    
    # To handle the state (current position) across the sequence of segments,
    # we can use a helper function or a reduction. 
    # However, since N is very small (N <= 6), we can simulate the path.
    
    def calculate_time(perm, dirs):
        # Starting point
        curr_pos = (0, 0)
        total_time = 0.0
        
        # We use a list comprehension to simulate the sequence and sum the results.
        # Because we need the 'end' of the previous segment to be the 'start' 
        # of the next move, we can't use a simple map. 
        # We will use a custom reduction-like approach via a helper.
        
        def folder(state, idx):
            pos, time = state
            direction = dirs[idx] # This is wrong because dirs is mapped to segment index, not perm index
            # Wait, dirs should be mapped to the segment index.
            # Let's redefine: dirs is a tuple where dirs[i] is the direction for segment i.
            
            seg = segments[perm[idx]]
            # Determine start and end points based on direction
            # If dir == 0: start=A, end=C. If dir == 1: start=C, end=A.
            # But the direction is specific to the segment, not the order.
            # Actually, it's easier to say: for the i-th segment in the permutation,
            # we choose whether to go from p1 to p2 or p2 to p1.
            return (None, None) # Placeholder

        return 0

    # Correcting the logic to avoid loops and recursion:
    # We can use a list comprehension to calculate the cost of a specific 
    # permutation and a specific set of directions.
    
    # Let's redefine the approach:
    # For a fixed permutation P and a fixed set of directions D (where D[i] is the direction for P[i]):
    # Time = dist((0,0), start_1)/S + len_1/T + dist(end_1, start_2)/S + len_2/T ...
    
    # To calculate this without loops, we can use a list comprehension to get the 
    # sequence of points and then sum the distances.
    
    def get_total_time(p, d):
        # p: permutation of indices
        # d: tuple of 0/1 indicating if we start at endpoint A or C for the segment p[i]
        
        # Points sequence: (start1, end1, start2, end2, ...)
        points = [
            (segments[p[i]][d[i]], segments[p[i]][1 - d[i]])
            for i in range(N)
        ]
        
        # Flatten the points to a sequence: start1, end1, start2, end2...
        flat_points = [pt for pair in points for pt in pair]
        # Add origin to the start
        path = [(0, 0)] + flat_points
        
        # Travel times:
        # Even indices in path (0, 2, 4...): moving to start of segment (Speed S)
        # Odd indices in path (1, 3, 5...): printing the segment (Speed T)
        
        # Move times (S): dist(path[0], path[1]), dist(path[1], path[2]) is WRONG.
        # Correct: 
        # Move 0: path[0] -> path[1] (S)
        # Print 1: path[1] -> path[2] (T)
        # Move 1: path[2] -> path[3] (S)
        # Print 2: path[3] -> path[4] (T)
        
        # Move times: dist(path[2i], path[2i+1]) / S
        # Print times: dist(path[2i+1], path[2i+2]) / T
        
        move_times = [dist(path[2*i], path[2*i+1]) / S for i in range(N)]
        print_times = [dist(path[2*i+1], path[2*i+2]) / T for i in range(N)]
        
        return sum(move_times) + sum(print_times)

    # We need to iterate over all permutations and all 2^N direction combinations.
    # Since we can't use loops, we use a generator expression inside min().
    
    # Note: directions should be tied to the position in the permutation, not the segment ID.
    # So for any permutation, there are 2^N ways to traverse the segments.
    
    ans = min(
        get_total_time(p, d)
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()