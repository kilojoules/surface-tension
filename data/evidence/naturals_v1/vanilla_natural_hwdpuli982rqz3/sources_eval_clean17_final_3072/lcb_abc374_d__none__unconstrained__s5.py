import sys
from itertools import permutations
from math import sqrt

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # Map input to variables
    # N: int, S: float, T: float
    # Segments: list of tuples ((ax, ay), (cx, cy))
    N = int(input_data[0])
    S = float(input_data[1])
    T = float(input_data[2])
    
    coords = list(map(float, input_data[3:]))
    segments = [((coords[i*4], coords[i*4+1]), (coords[i*4+2], coords[i*4+3])) for i in range(N)]
    
    # Precompute lengths of segments
    lengths = [sqrt((s[0][0]-s[1][0])**2 + (s[0][1]-s[1][1])**2) for s in segments]
    
    # Helper to calculate distance between two points
    dist = lambda p1, p2: sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    # We need to decide the order of segments and the direction of each segment.
    # Order: N! possibilities. Direction: 2^N possibilities.
    # Total complexity: N! * 2^N. For N=6, this is 720 * 64 = 46,080, which is very small.
    
    # Generate all permutations of segment indices
    all_orders = list(permutations(range(N)))
    
    # For each order, we use dynamic programming or recursion to find the min time.
    # Since we can't use loops for logic flow (though we can for comprehensions), 
    # we can use a recursive function or a bitmask DP.
    # Given the constraints and the "no loop" requirement for logic, 
    # we can evaluate all 2^N direction combinations using a list comprehension.
    
    def evaluate_path(order, directions):
        # directions is a tuple of 0 or 1. 
        # 0 means start at s[0], end at s[1]. 1 means start at s[1], end at s[0].
        
        # Generate the sequence of points: (start1, end1, start2, end2, ...)
        points = [
            (segments[order[i]][0] if directions[i] == 0 else segments[order[i]][1],
             segments[order[i]][1] if directions[i] == 0 else segments[order[i]][0])
            for i in range(N)
        ]
        
        # Calculate total time:
        # Time = dist(origin, p0_start)/S + length0/T + dist(p0_end, p1_start)/S + length1/T ...
        
        # Initial move from (0,0)
        start_move = dist((0, 0), points[0][0]) / S
        
        # Printing times
        print_times = sum([lengths[i] / T for i in range(N)])
        
        # Intermediate moves
        inter_moves = sum([dist(points[i][1], points[i+1][0]) / S for i in range(N-1)])
        
        return start_move + print_times + inter_moves

    # Use a nested comprehension to find the minimum time across all permutations and directions
    # 2**N directions can be represented by range(2**N) and bit shifting
    ans = min([
        evaluate_path(order, tuple((j >> i) & 1 for i in range(N)))
        for order in all_orders
        for j in range(1 << N)
    ])
    
    print(f"{ans:.20f}")

if __name__ == "__main__":
    solve()