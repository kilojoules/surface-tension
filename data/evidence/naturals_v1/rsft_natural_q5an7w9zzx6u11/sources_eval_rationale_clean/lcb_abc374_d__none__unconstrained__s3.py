import sys
import math
from itertools import permutations, product

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
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Precompute lengths of segments to avoid redundant calculations
    seg_lengths = [dist(s[0], s[1]) for s in segments]

    # We need to try all permutations of segments and all directions for each segment.
    # A direction is represented by 0 (start -> end) or 1 (end -> start).
    
    # Generate all permutations of segment indices
    all_perms = permutations(range(N))
    # Generate all possible direction combinations (2^N)
    all_dirs = product([0, 1], repeat=N)

    # To avoid nested loops, we use a generator expression.
    # For a fixed permutation and direction set:
    # Total time = sum(dist(current_pos, next_start) / S) + sum(seg_length / T)
    
    # The sum of (seg_length / T) is constant regardless of order.
    constant_time = sum(l / T for l in seg_lengths)

    # We seek to minimize the travel time between segments.
    # We use a helper function to calculate travel time for a specific sequence.
    def calculate_travel_time(perm, dirs):
        # Create the sequence of points visited.
        # Each segment i is printed from p_start to p_end.
        # points will be [(start1, end1), (start2, end2), ...]
        ordered_segs = [
            (segments[i][dirs[p]] if dirs[p] == 0 else segments[i][1],
             segments[i][1-dirs[p]] if dirs[p] == 0 else segments[i][0])
            for p, i in zip(range(N), perm)
        ]
        
        # The travel distances are:
        # (0,0) -> start1
        # end1 -> start2
        # end2 -> start3 ...
        # We use a list comprehension to get the pairs of (end_prev, start_curr)
        travel_pairs = [
            ((0, 0), ordered_segs[0][0])
        ] + [
            (ordered_segs[i][1], ordered_segs[i+1][0])
            for i in range(N - 1)
        ]
        
        return sum(dist(p1, p2) for p1, p2 in travel_pairs) / S

    # Since we cannot use loops, we use a generator expression inside min().
    # We need to iterate over all permutations and all direction combinations.
    # Note: dirs is mapped to the permutation. 
    # Actually, it's easier to think: for a permutation, each segment has 2 choices.
    
    # To avoid the O(N! * 2^N) being too slow in a single expression, 
    # we can optimize: for a fixed permutation, the choice of direction for segment i
    # only depends on the end of segment i-1 and the start of segment i+1.
    # However, with N=6, N! * 2^N = 720 * 64 = 46,080, which is small enough.

    # We use a nested generator:
    # For each permutation, we try all 2^N direction combinations.
    # But wait, the directions are tied to the segments in the permutation.
    # Let's redefine: dirs[j] is the direction of the j-th segment in the permutation.
    
    ans = min(
        sum(
            dist(
                (0, 0) if j == 0 else (
                    segments[perm[j-1]][1 if d[j-1] == 0 else 0] 
                    if d[j-1] == 0 else segments[perm[j-1]][0]
                ), # This logic is getting complex, let's simplify.
                # Let's use a helper to get the endpoints based on direction.
                # If d=0: (seg[0], seg[1]), if d=1: (seg[1], seg[0])
                # The point is: we need the start and end of the current segment.
                # Let's use a more direct approach.
                None # Placeholder
            ) for j in range(N)
        ) for perm in permutations(range(N)) for d in product([0, 1], repeat=N)
    )
    
    # Let's rewrite the travel time calculation clearly using a list comprehension.
    # For a given permutation 'p' and direction vector 'd':
    # Segment j (in permutation) is segments[p[j]].
    # Its endpoints are A = segments[p[j]][0], B = segments[p[j]][1].
    # If d[j] == 0, it's printed A -> B. If d[j] == 1, it's printed B -> A.
    
    # To avoid the complexity in the generator, we can pre-calculate 
    # the start and end points for each segment/direction pair.
    # pts[seg_idx][dir] = (start_pt, end_pt)
    pts = [
        [(segments[i][0], segments[i][1]), (segments[i][1], segments[i][0])]
        for i in range(N)
    ]

    # Now the travel time is:
    # dist((0,0), pts[p[0]][d[0]][0]) / S
    # + sum(dist(pts[p[j]][d[j]][1], pts[p[j+1]][d[j+1]][0]) / S for j in range(N-1))
    
    # We can use a generator expression to find the minimum travel time.
    min_travel = min(
        (
            dist((0, 0), pts[p[0]][d[0]][0]) / S +
            sum(dist(pts[p[j]][d[j]][1], pts[p[j+1]][d[j+1]][0]) / S for j in range(N-1))
        )
        for p in permutations(range(N))
        for d in product([0, 1], repeat=N)
    )

    print(f"{min_travel + constant_time:.20f}")

if __name__ == "__main__":
    solve()