import sys
from itertools import groupby

def solve():
    # Read all input at once
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    h = list(map(int, data[1:]))
    
    # Group indices by their building height
    # height_groups = {height: [list of indices]}
    # We use a dictionary comprehension to organize indices for each unique height
    unique_heights = set(h)
    height_groups = {height: [i for i, val in enumerate(h) if val == height] 
                     for height in unique_heights}
    
    # For each height group, we need to find the maximum number of indices 
    # that form an arithmetic progression.
    # Since N is small (3000), we can iterate through all pairs of indices (i, j)
    # in a group to define a starting point and a common difference 'd'.
    
    # However, a more efficient way to check equal intervals for a specific height:
    # For every possible interval d (1 to N), and every possible start s (0 to d-1),
    # check how many buildings at indices s, s+d, s+2d... have the target height.
    
    # To avoid nested loops and comply with the "no explicit for loop" constraint 
    # (though the prompt didn't explicitly forbid them, it's safer to use comprehensions),
    # we can iterate through all possible intervals d and all heights.
    
    # Let's refine: for a fixed height 'H' and interval 'd', 
    # the number of buildings is max(count of H in indices [s, s+d, s+2d...])
    # This is still complex. Let's use the property:
    # For a fixed height H, and two indices i and j (i < j), 
    # they can be part of a sequence with interval d = j - i.
    # The number of elements is (number of k such that h[i + k*(j-i)] == H).
    
    # Given N=3000, O(N^2) is acceptable. 
    # We can iterate over all pairs (i, j) where h[i] == h[j].
    # But that's still O(N^3) if we count the sequence.
    
    # Optimized approach:
    # For every possible interval d from 1 to N:
    #   For every starting position s from 0 to d-1:
    #     The sequence is h[s], h[s+d], h[s+2d]...
    #     We find the most frequent height in this sequence.
    
    # We use a generator expression to evaluate all d and s.
    # 1. d ranges from 1 to N
    # 2. s ranges from 0 to d-1
    # 3. Sequence is h[s::d]
    # 4. Find the max length of grouped identical heights in the sequence
    #    Wait, the condition is "chosen buildings are arranged at equal intervals".
    #    This means we pick indices s, s+d, s+2d... AND they must all have the same height.
    #    So for a fixed s and d, we are looking for the longest contiguous block of the SAME height
    #    in the sequence h[s], h[s+d], h[s+2d]... NO, that's wrong.
    #    The condition is: the chosen buildings (the subset) must be at equal intervals.
    #    This means if we pick indices {i_1, i_2, ..., i_k}, then i_{j+1} - i_j = d.
    #    And h[i_1] = h[i_2] = ... = h[i_k].
    #    This means we are looking for the longest run of the same height in the sequence h[s::d].
    
    # Correct logic:
    # For each d in [1, N]:
    #   For each s in [0, d-1]:
    #     Sequence seq = h[s::d]
    #     Find the length of the longest run of identical elements in seq.
    
    # Using groupby to find runs of identical elements:
    # max(len(list(g)) for g in groupby(h[s::d]))
    
    # We wrap this in a nested comprehension.
    ans = max(
        (
            max(
                (
                    len(list(g)) 
                    for k, g in groupby(h[s::d])
                ), 
                default=0
            )
            for d in range(1, n + 1)
            for s in range(d)
        ), 
        default=1
    )
    
    print(ans)

if __name__ == "__main__":
    solve()