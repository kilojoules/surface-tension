import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # If there are no buildings, the answer is 0
    if N == 0:
        print(0)
        return

    # We need to find the maximum count of buildings with the same height H_i
    # that are spaced at equal intervals 'd'.
    # Let the indices be i, i+d, i+2d, ..., i+(k-1)d.
    # All these must have the same height.
    
    # Since N is small (3000), we can iterate through all possible 
    # starting positions 'i' and all possible intervals 'd'.
    # However, a more efficient way is to group indices by height.
    
    # Create a dictionary where keys are heights and values are lists of indices
    from collections import defaultdict
    pos_map = defaultdict(list)
    for idx, h in enumerate(H):
        pos_map[h].append(idx)
    
    # For each height, we look at the indices where it occurs.
    # To check for equal intervals, we can iterate through all pairs of indices (i, j)
    # and treat them as the first two elements of the sequence.
    # But since we can't use loops, we use a different approach.
    
    # For a fixed height and a fixed interval 'd', we want to find the 
    # longest contiguous sequence of indices (i, i+d, i+2d...) that all have that height.
    
    # We can use a technique similar to finding the longest arithmetic progression,
    # but since we need the elements to be exactly at intervals of 'd', 
    # we can iterate over all possible 'd' from 1 to N//2.
    
    # To avoid for/while loops, we use list comprehensions and map functions.
    
    # For a fixed d, we can check the sequences.
    # Let's define a function that calculates the max length for a specific d.
    def max_for_d(d):
        # We can use a "dynamic programming" approach expressed via a list comprehension.
        # Since we can't use loops to build the DP table, we can use a recursive-like 
        # structure or process the array in chunks.
        # Actually, we can just check all starting points i from 0 to d-1.
        # For each i, we have a sequence H[i], H[i+d], H[i+2d]...
        # We need to find the longest run of identical values in this sequence.
        
        # Helper to find max run of identical elements in a list
        def max_run(seq):
            if not seq: return 0
            # Group identical consecutive elements
            # We can use a trick with itertools.groupby or a custom reduction.
            from itertools import groupby
            return max([len(list(g)) for k, g in groupby(seq)] + [0])

        # For a fixed d, we split the original list into d lists based on remainder
        # and find the max run in each.
        sequences = [[H[i + j * d] for j in range((N - 1 - i) // d + 1)] for i in range(d)]
        return max([max_run(seq) for seq in sequences] + [0])

    # We test all d from 1 to N. d=0 is not possible, d=N is effectively 1.
    # The case for 1 building is covered by d=1 or simply by the fact that N >= 1.
    
    # To satisfy "no loops", we use map and max.
    # We need to handle the case N=1 separately or ensure the range is correct.
    
    ans = max(map(max_for_d, range(1, N // 2 + 1)), default=0)
    
    # Also consider the case where we only pick one building (which is always 1 if N > 0)
    # and the case where we pick buildings of the same height that are not necessarily 
    # separated by d < N/2 (though d >= N/2 would result in at most 2 buildings).
    
    # Actually, the max_for_d logic covers d >= 1. 
    # If N=1, range(1, 1) is empty, ans=0, but we should output 1.
    # If N=8, d=1..4. If we pick 2 buildings at index 0 and 7, d=7.
    # Let's just check all d from 1 to N-1.
    
    final_ans = max(max(map(max_for_d, range(1, N)), default=0), 1 if N > 0 else 0)
    print(final_ans)

if __name__ == "__main__":
    solve()