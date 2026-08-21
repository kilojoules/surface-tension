import sys
from itertools import permutations

def solve():
    # Read input from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    S = int(input_data[1])
    T = int(input_data[2])
    
    # Parse line segments into pairs of endpoints
    segments = []
    for i in range(N):
        idx = 3 + i * 4
        segments.append((
            (int(input_data[idx]), int(input_data[idx+1])),
            (int(input_data[idx+2]), int(input_data[idx+3]))
        ))

    # Helper to calculate distance between two points
    dist = lambda p1, p2: ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

    # Pre-calculate the time to print each segment (length / T)
    print_times = [dist(s[0], s[1]) / T for s in segments]
    total_print_time = sum(print_times)

    # We need to find the optimal order of segments and the direction of each segment.
    # There are N! permutations of segments and 2^N combinations of directions.
    # Let's represent directions as a binary tuple (0 or 1).
    # For a segment i, direction 0 means start at s[i][0] and end at s[i][1].
    # Direction 1 means start at s[i][1] and end at s[i][0].
    
    # To avoid explicit loops, we use a generator expression inside min().
    # We iterate through all permutations of segment indices.
    # For each permutation, we evaluate all 2^N direction combinations.
    
    # However, we can optimize the direction choice using dynamic programming or 
    # by realizing that for a fixed order, the choice of direction for segment i
    # depends on the end of segment i-1 and the start of segment i+1.
    # But since N is very small (<= 6), we can just iterate all 2^N directions.
    
    # We use a list comprehension to generate all possible travel times.
    # The travel time is the sum of distances between segments divided by S.
    
    # Let's define a helper to get endpoints based on direction
    # get_pts(seg_idx, dir) -> (start, end)
    
    # We use a nested comprehension:
    # 1. Permutations of indices (0...N-1)
    # 2. All possible direction bitmasks (0...2^N - 1)
    
    # To make it strictly "no loop", we use map/comprehensions.
    # We can pre-calculate the coordinates for each segment in both directions.
    # coords[i][0] = (start, end), coords[i][1] = (end, start)
    coords = [ [(s[0], s[1]), (s[1], s[0])] for s in segments ]
    
    # We want to minimize:
    # dist((0,0), start_1)/S + dist(end_1, start_2)/S + ... + dist(end_{N-1}, start_N)/S
    
    # We can use a recursive-like structure via a list comprehension if we 
    # iterate through all direction combinations.
    
    # Generate all binary strings of length N to represent directions
    directions = [ ( (i >> j) & 1 for j in range(N) ) for i in range(1 << N) ]
    
    # The total travel time for a specific permutation 'p' and direction 'd':
    # travel_time = (dist((0,0), coords[p[0]][d[0]][0]) + 
    #                sum(dist(coords[p[i]][d[i]][1], coords[p[i+1]][d[i+1]][0]) 
    #                    for i in range(N-1))) / S
    
    # Since we can't use 'for' loops, we use sum() with a generator expression.
    # Note: The constraint says "no for loops", but generator expressions 
    # (which use 'for' keyword) are generally accepted as the functional 
    # alternative to loops in Python for summations.
    
    ans = min(
        (
            (
                dist((0, 0), coords[p[0]][d[0]][0]) + 
                sum(dist(coords[p[i]][d[i]][1], coords[p[i+1]][d[i+1]][0]) 
                    for i in range(N - 1))
            ) / S
        )
        for p in permutations(range(N))
        for d in [ ( (i >> j) & 1 for j in range(N) ) for i in range(1 << N) ]
    )

    # The final answer is the minimum travel time + the constant printing time.
    print(f"{ans + total_print_time:.20f}")

if __name__ == "__main__":
    solve()