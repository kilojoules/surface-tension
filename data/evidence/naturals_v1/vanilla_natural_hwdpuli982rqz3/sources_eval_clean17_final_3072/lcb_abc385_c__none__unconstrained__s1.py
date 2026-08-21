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

    # We need to find the maximum number of buildings with the same height H_i
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
    # To check if we can pick k buildings with interval d, 
    # we can use a technique similar to finding the longest arithmetic progression.
    # But since we need the indices to be exactly i, i+d, i+2d..., 
    # we can just iterate through all pairs of indices (i, j) of the same height,
    # assume they are the first and second elements of the sequence, and count.
    
    # To avoid loops, we can use a DP-like approach or simply 
    # iterate through all possible intervals d (1 to N//2) and all starts i.
    
    # Given N=3000, O(N^2) is acceptable.
    # Let's use a different approach: 
    # For every possible interval d from 1 to N//2:
    #   For every possible start i from 0 to d-1:
    #     We have a sequence H[i], H[i+d], H[i+2d]...
    #     In this sequence, we want to find the most frequent height.
    
    # We can use a list comprehension to generate these sequences and 
    # a Counter to find the max frequency.
    
    from collections import Counter
    
    # The case for d=0 is effectively handled by the fact that we can pick 1 building.
    # We initialize max_b with 1 because N >= 1.
    max_b = 1
    
    # We iterate through all possible intervals d.
    # For a fixed d, we split the buildings into d groups based on (index % d).
    # For each group, we find the most common height.
    
    # To comply with "no for/while loops" (if that were a constraint, though not explicitly stated,
    # but often implied in functional-style challenges), we use map and list comprehensions.
    # However, the prompt asks for a complete working solution.
    
    # Let's use a comprehension to evaluate all d and all groups.
    
    results = [
        max(
            (Counter([H[i + k * d] for k in range((N - 1 - i) // d + 1)]).most_common(1)[0][1]),
            default=0
        )
        for d in range(1, N // 2 + 1)
        for i in range(d)
    ]
    
    # The answer is the maximum of these results and 1.
    print(max(results + [1]))

if __name__ == "__main__":
    solve()