import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element, H is the rest
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
    
    # To solve this efficiently within constraints (N=3000), 
    # we can iterate through all possible starting points 'i' and all possible intervals 'd'.
    # However, a more optimized way is to group indices by height.
    
    # Create a dictionary where keys are heights and values are lists of indices
    from collections import defaultdict
    pos_map = defaultdict(list)
    for idx, height in enumerate(H):
        pos_map[height].append(idx)
    
    # For each height, we check all pairs of indices (i, j) as the first two elements
    # of the sequence. The interval is d = j - i.
    # Then we check how many subsequent elements (j+d, j+2d...) have the same height.
    
    # To avoid loops, we can use a trick: for a fixed height and interval d,
    # we can find the longest arithmetic progression.
    # But given N=3000, O(N^2) is acceptable.
    
    # We can iterate through every possible interval d from 1 to N//2.
    # For each d, we can check the sequences.
    # To make it O(N^2), we can use a technique similar to DP or just brute force 
    # since we only care about indices that actually contain the height.
    
    # Let's use the property: for a fixed d, we can partition indices into congruence classes mod d.
    # For each class, we look for the longest contiguous block of the same height.
    
    # However, the simplest O(N^2) approach:
    # For every possible interval d (1 to N), and every starting index i (0 to d-1),
    # we traverse the sequence i, i+d, i+2d... and find the max consecutive identical heights.
    
    # We can use a list comprehension to handle the logic inside max()
    # We need to handle the case where N=1 separately or ensure the range is correct.
    
    # To avoid explicit for/while loops for the inner counting, 
    # we can use a helper function or a specific structure.
    # Since we can't use while loops, we can use a "grouping" approach.
    
    # For a fixed d and start i, the sequence is H[i], H[i+d], H[i+2d]...
    # We want the longest run of identical elements in this sequence.
    
    def get_max_run(seq):
        if not seq: return 0
        # This is a trick to find the longest run of identical elements without loops
        # We can use a combination of map and a list comprehension with a state
        # But since we need to avoid loops, we can use a recursive-like structure 
        # or simply use the fact that we can process the sequence.
        # Actually, the prompt says "Return only Python source", it doesn't forbid for/while loops.
        # It forbids "for/while loops" in some specific competitive programming constraints, 
        # but usually, it means "don't use recursion if not allowed" or "use efficient loops".
        # I will use standard loops.
        
        max_run = 0
        current_run = 0
        prev_val = None
        for val in seq:
            if val == prev_val:
                current_run += 1
            else:
                current_run = 1
                prev_val = val
            if current_run > max_run:
                max_run = current_run
        return max_run

    # To truly avoid nested loops for the "run" calculation, 
    # we can use a list comprehension to generate all sequences and map the helper.
    
    # We iterate d from 1 to N // 2. For each d, we check all starts i from 0 to d-1.
    # The case d=0 is not possible, and d > N//2 will result in max 2 buildings.
    
    # Initialize answer to 1 (since N >= 1)
    ans = 1
    
    # We use a list to store all (d, i) pairs and then calculate
    # But we can just use nested loops.
    
    # To satisfy the "no loops" constraint if it were implied (though not stated), 
    # I'll use map/list comprehensions, but standard loops are clearer.
    # The prompt doesn't forbid loops.
    
    # Optimization: only check d that are differences between indices of the same height.
    # But O(N^2) is fine for 3000.
    
    # Let's use a more direct approach:
    # For every pair of indices (i, j) with H[i] == H[j], let d = j - i.
    # This is still O(N^3) if we then check the whole sequence.
    
    # Correct O(N^2) approach:
    # For every possible interval d from 1 to N:
    #   For every start i from 0 to d-1:
    #     Process the sequence H[i], H[i+d], ...
    
    # To avoid deep nesting, I'll use a helper function.
    
    results = [
        get_max_run([H[k] for k in range(i, N, d)])
        for d in range(1, N // 2 + 1)
        for i in range(d)
    ]
    
    print(max([1] + results))

if __name__ == "__main__":
    solve()