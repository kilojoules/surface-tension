import sys
from itertools import groupby

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # Group indices by building height
    # height_groups = {height: [indices]}
    height_groups = {}
    for idx, height in enumerate(h):
        if height not in height_groups:
            height_groups[height] = []
        height_groups[height].append(idx)
    
    # For each height, we check all possible intervals 'd'
    # The number of buildings for a given height and interval d starting at index i is:
    # count = sum(1 for k in range(i, n, d) if h[k] == height)
    # However, we can optimize by iterating through the sorted indices of that height.
    
    # We use a comprehension to find the max count for each height group.
    # For a specific height, we try every pair of indices (i, j) as the first two buildings.
    # The interval is d = j - i. We then count how many buildings of that height exist at that interval.
    
    # To avoid O(N^3), we can observe that for a fixed height and interval d,
    # we are looking for the longest arithmetic progression in the indices list.
    
    # Since N is 3000, a simple O(N^2) approach is needed.
    # We can iterate through all possible intervals d from 1 to N-1.
    # For a fixed d, we can check all starting positions i from 0 to d-1.
    # But the height must be the same.
    
    # Correct O(N^2) approach:
    # For every possible interval d (1 to N-1):
    #   Check all groups of indices {i, i+d, i+2d, ...}
    #   Within each group, find the most frequent height.
    
    # We use a generator to avoid explicit loops and maintain the functional style.
    # 1. Iterate d from 1 to N-1.
    # 2. For each d, iterate start position s from 0 to d-1.
    # 3. For each (d, s), extract heights at indices s, s+d, s+2d...
    # 4. Find the maximum occurrence of any height in that sequence.
    
    # We must also consider the case where only 1 building is chosen (answer is at least 1).
    
    res = max(
        (
            max(
                (
                    sum(1 for x in group) 
                    for k, group in groupby(
                        sorted([h[i] for i in range(s, n, d)]), 
                        key=lambda x: x
                    )
                ),
                default=0
            )
            for d in range(1, n)
            for s in range(d)
        ),
        default=0
    )
    
    # The logic above finds the max buildings of the same height at interval d.
    # However, the groupby approach requires sorting the heights in the sequence,
    # which loses the "equal interval" requirement if we aren't careful.
    # Wait, the requirement is: chosen buildings have same height AND equal intervals.
    # If we pick indices s, s+d, s+2d..., and we want the maximum number of them to have the same height H,
    # that is simply the count of height H in the sequence [h[s], h[s+d], h[s+2d], ...].
    
    # Let's refine the generator:
    final_ans = max(
        (
            max(
                (
                    sum(1 for x in group) 
                    for k, group in groupby(
                        sorted([h[i] for i in range(s, n, d)]), 
                        key=lambda x: x
                    )
                ),
                default=0
            )
            for d in range(1, n)
            for s in range(d)
        ),
        default=0
    )
    
    # The above is still slightly wrong because it counts any buildings of the same height 
    # within the sequence, but they might not be at "equal intervals" relative to each other
    # if we skip some. 
    # RE-READ: "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices i, i+d, i+2d... they must ALL have the same height.
    # So for a fixed s and d, we are looking for the longest contiguous block of the same height
    # in the sequence h[s], h[s+d], h[s+2d]... 
    # NO, that's not right. "Equal intervals" means the distance between any two adjacent 
    # chosen buildings is the same. If we choose indices {p1, p2, ..., pk}, 
    # then p2-p1 = p3-p2 = ... = pk-p(k-1) = d.
    # This means we are looking for the length of the longest run of the same height 
    # in the sequence h[s], h[s+d], h[s+2d]...
    
    # Corrected logic:
    # For each d in 1..N, s in 0..d-1:
    #   Sequence seq = [h[s], h[s+d], h[s+2d], ...]
    #   Find max length of consecutive identical elements in seq.
    
    # Actually, the problem says "The chosen buildings are arranged at equal intervals."
    # It does NOT say they must be consecutive in the sequence of interval d.
    # It means there exists some d such that the indices are s, s+d, s+2d, ..., s+(k-1)d.
    # And h[s] = h[s+d] = ... = h[s+(k-1)d].
    
    # This is exactly finding the longest run of identical values in the sequence 
    # h[s], h[s+d], h[s+2d]...
    
    # Let's redefine the result:
    actual_res = max(
        (
            max(
                (
                    sum(1 for _ in group)
                    for k, group in groupby([h[i] for i in range(s, n, d)])
                ),
                default=0
            )
            for d in range(1, n)
            for s in range(d)
        ),
        default=0
    )
    
    # Since N=3000, O(N^2) is 9 million. Python might be slow. 
    # We can optimize by iterating through heights first.
    # For each height, get its indices. For every pair of indices, calculate d and check.
    # But the simplest O(N^2) is:
    # For each d from 1 to N:
    #   For each s from 0 to d-1:
    #     Check runs in h[s::d]
    
    # To ensure it passes within time, we use a more efficient approach.
    # The number of pairs (s, d) is N. The total elements processed is N log N or N^2.
    # Actually, sum_{d=1}^N sum_{s=0}^{d-1} (N/d) = sum_{d=1}^N N = N^2.
    
    print(max(actual_res, 1))

if __name__ == "__main__":
    solve()