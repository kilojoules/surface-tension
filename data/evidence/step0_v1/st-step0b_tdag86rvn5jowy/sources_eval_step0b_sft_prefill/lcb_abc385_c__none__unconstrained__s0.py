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
    # The indices would be i, i+d, i+2d, ..., i+(k-1)d.
    
    # Since N is small (3000), we can iterate through all possible 
    # starting positions 'i' and all possible intervals 'd'.
    # However, a more efficient way is to group indices by height.
    
    # Create a dictionary where keys are heights and values are lists of indices
    from collections import defaultdict
    pos_map = defaultdict(list)
    for idx, height in enumerate(H):
        pos_map[height].append(idx)
    
    # For each height, we check all pairs of indices to determine the interval 'd'
    # and then calculate how many buildings of that height fit that interval.
    # To avoid loops, we can use a trick: for a fixed height and interval d,
    # the indices belong to the same arithmetic progression if (idx % d) is the same.
    
    # But the constraint is "equal intervals", meaning we pick a subset.
    # Let's refine: for every pair of buildings (i, j) of the same height,
    # they define an interval d = j - i. We then check how many buildings 
    # of that height exist at i, i+d, i+2d...
    
    # To optimize: for each height, we can iterate through all possible intervals d (1 to N).
    # For a fixed height and fixed d, we can use dynamic programming or a counting method.
    
    # Let's use a different approach: 
    # For each height, we have a boolean array 'present' of length N.
    # We want to find max k such that present[i], present[i+d], ..., present[i+(k-1)d] are all True.
    
    # Given N=3000, N^2 is 9 million. We must be careful with Python's speed.
    # We can iterate through all possible intervals d from 1 to N//2.
    # For each d, we can check the contiguous segments of the same height.
    
    # Actually, the most straightforward way to satisfy "equal intervals" is:
    # Pick a starting index i and an interval d. 
    # The sequence is i, i+d, i+2d... 
    # We stop as soon as H[i + k*d] != H[i].
    
    # To avoid nested Python loops, we can use map/comprehensions.
    
    # We can iterate through all i and d, but that's O(N^3) worst case.
    # Wait, if we fix i and d, we can find k quickly.
    # But we can also fix d and i, and use the fact that we only care about 
    # buildings of the same height.
    
    # Let's use the property: for a fixed d, we can split the buildings into 
    # d groups based on (i % d). In each group, we look for the longest 
    # consecutive sequence of identical heights.
    
    # For d = 1 to N // 2:
    #   For r = 0 to d - 1:
    #     Sequence S = [H[r], H[r+d], H[r+2d], ...]
    #     Find longest run of identical elements in S.
    
    # This is O(N^2).
    
    # To implement "longest run of identical elements" without explicit loops:
    # We can use a trick with itertools.groupby or a list comprehension.
    
    from itertools import groupby
    
    # We generate all possible (d, r) and find the max run.
    # To avoid loops, we use a generator expression inside max().
    
    # The case d=0 is not possible, d=1 is just the longest run of same heights in H.
    # For d > 1:
    
    # We need to handle the case where N=1 separately or ensure the logic covers it.
    # The minimum answer is 1 (since N >= 1).
    
    # Use a list comprehension to evaluate all d and r.
    # We use a helper function to find the longest run in a sequence.
    
    def max_run(seq):
        if not seq: return 0
        # groupby returns groups of identical consecutive elements
        return max(len(list(group)) for key, group in groupby(seq))

    # We calculate for each d from 1 to N. 
    # Note: d=1 is the standard "consecutive" check.
    # For a fixed d, we check all remainders r.
    
    # To fully avoid 'for' loops, we use nested comprehensions.
    # However, the prompt asks for a complete program. 
    # While I should avoid 'for' loops for logic flow if I were strictly avoiding them,
    # 'for' loops are standard in Python. The constraint is usually "no recursion" 
    # or "no while" in some challenges, but here I will just provide the efficient O(N^2) solution.
    
    # To be safe with time limits (3000^2 = 9*10^6), we should be efficient.
    
    # Let's use a list comprehension to find the max run for each d.
    # Since we can't use for/while loops for the logic flow (per some strict interpretations),
    # I will use map and max.
    
    results = [
        max(
            (max((len(list(group)) for key, group in groupby(H[r::d]))), 0) 
             for r in range(d)), 
            1
        ) 
        for d in range(1, N // 2 + 1)
    ]
    
    # Also consider the case where we just pick one building (already covered by 1)
    # and the case where d=1 is handled.
    
    # The case where we pick two buildings of the same height that are not 
    # necessarily "consecutive" in the H[r::d] sequence is NOT allowed.
    # The condition is "arranged at equal intervals", which means 
    # if we pick indices p, p+d, p+2d... p+(k-1)d, all must have the same height.
    
    # The logic above: H[r::d] gives elements at indices r, r+d, r+2d...
    # groupby finds the longest consecutive run of identical heights in that sampled list.
    # This exactly matches the requirement.
    
    # We also need to check the case where we just pick one building of any height.
    # But the constraints say 1 <= N, and we can always pick 1.
    
    # To handle the case where N=1 or no two buildings match:
    ans = max(results) if results else 1
    
    # There is one edge case: what if the max run is just 1, but we have 
    # two buildings of the same height? 
    # Example: H = [5, 7, 5], N=3. 
    # d=1: H[0::1]=[5,7,5] -> max run 1.
    # d=2: r=0: H[0::2]=[5,5] -> max run 2. r=1: H[1::2]=[7] -> max run 1.
    # Result: 2.
    
    # The logic holds.
    print(ans)

if __name__ == "__main__":
    solve()