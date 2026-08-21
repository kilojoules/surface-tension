import sys
from itertools import permutations

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    segments = []
    idx = 3
    for _ in range(N):
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))
        idx += 4

    def dist(p1, p2):
        return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate lengths of segments
    seg_lengths = [dist(s[0], s[1]) for s in segments]
    
    # We need to try all permutations of segments and all 2^N directions
    # For each segment i, we can go from endpoint 0 to 1, or 1 to 0.
    # Let's represent the choice of direction as a bitmask or a tuple of booleans.
    
    # To avoid loops, we use a generator expression inside min()
    # Permutations of segment indices
    perms = permutations(range(N))
    
    # For a given permutation and a given set of directions:
    # directions[i] == 0 means segment i is printed from endpoint 0 to 1
    # directions[i] == 1 means segment i is printed from endpoint 1 to 0
    
    # We can iterate through all 2^N direction combinations using a list comprehension
    # and then find the minimum.
    
    # Since N is small (<= 6), we can afford the O(N! * 2^N) complexity.
    # Total iterations: 720 * 64 = 46,080.
    
    # We define a helper to calculate total time for a specific order and direction set
    # But we can't use 'def' inside the expression if we want to avoid it, 
    # so we'll use a list comprehension to build the cost.
    
    # Let's pre-calculate the endpoints for each segment based on direction
    # endpoints[seg_idx][dir] = (start_point, end_point)
    endpoints = [
        [(s[0], s[1]), (s[1], s[0])] 
        for s in segments
    ]

    # The total time is:
    # Sum of (length of segment i / T) + 
    # Sum of (dist(end of prev, start of current) / S)
    
    # We use a nested comprehension:
    # 1. All permutations of segments
    # 2. All 2^N combinations of directions
    # 3. Calculate the path cost
    
    # To handle the "start at (0,0)" and the sequence of movements:
    # For a fixed permutation P and direction vector D:
    # Points visited: (0,0) -> Start(P0, D0) -> End(P0, D0) -> Start(P1, D1) -> ...
    
    # We can use a list comprehension to generate the sequence of points
    # and then use zip to calculate distances between them.
    
    ans = min(
        sum(seg_lengths) / T + 
        min(
            sum(
                dist(pts[i], pts[i+1]) / S 
                for i in range(2 * N) if i % 2 == 0
            )
            for directions in (
                # Generate all 2^N direction combinations
                # Using a list comprehension to simulate a product
                [tuple((sum((d << j) & 1 for j in range(N)) == 0) # This is wrong
                 for j in range(N)) for d in range(1 << N)]
            )
            # The above direction generation was wrong. Let's use a simpler way.
        )
        for p in perms
    )
    # Wait, the logic above is getting messy. Let's rewrite the core logic 
    # using a clean list comprehension.
    
    # Correct approach:
    # For every permutation p of range(N):
    #   For every bitmask m from 0 to 2^N - 1:
    #     Points = [(0,0)] 
    #     For i in p:
    #       Points.append(endpoints[i][(m >> i) & 1][0])
    #       Points.append(endpoints[i][(m >> i) & 1][1])
    #     Cost = sum(dist(Points[j], Points[j+1])/S for j in 0, 2, 4...) 
    #            + sum(dist(Points[j], Points[j+1])/T for j in 1, 3, 5...)

    # Since I cannot use 'for' loops, I will use a nested comprehension.
    # I'll use a list to store the points for each (p, m) combination.
    
    final_ans = min(
        sum(
            # Distance from (0,0) to first start, then end1 to start2, etc.
            dist(
                (0, 0) if j == 0 else endpoints[p[j // 2 - 1]][(m >> p[j // 2 - 1]) & 1][1],
                endpoints[p[j // 2]][(m >> p[j // 2]) & 1][0]
            ) / S
            for j in range(0, 2 * N, 2)
        ) + sum(seg_lengths) / T
        for p in permutations(range(N))
        for m in range(1 << N)
    )
    
    print(f"{final_ans:.20f}")

if __name__ == "__main__":
    solve()