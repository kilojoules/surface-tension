import sys

def solve():
    # Read all input
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
    # height_map = {height: [index1, index2, ...]}
    from collections import defaultdict
    height_map = defaultdict(list)
    for idx, h in enumerate(H):
        height_map[h].append(idx)
    
    # For each height, we check all pairs of indices (i, j) as the first two 
    # buildings in the sequence. The interval is d = j - i.
    # Then we check how many subsequent buildings at that interval have the same height.
    
    # To avoid nested loops with while/for, we can use a trick:
    # For a fixed height and a fixed interval d, we can find the max length.
    # But the constraints allow O(N^2) if we are careful.
    
    # Let's use a different approach: 
    # For every possible interval d (1 to N//2), and every starting point i (0 to d-1),
    # we can treat the sequence H[i], H[i+d], H[i+2d]... as a string/list
    # and find the longest run of identical elements.
    
    # However, the "equal interval" doesn't mean we can't skip buildings.
    # Wait, the problem says "The chosen buildings are arranged at equal intervals."
    # This means if we pick indices p_1, p_2, ..., p_k, then p_{j+1} - p_j = d for all j.
    
    # Correct approach:
    # For every possible interval d from 1 to N-1:
    #   For every starting index i from 0 to d-1:
    #     Check the sequence H[i], H[i+d], H[i+2d]...
    #     In this sequence, find the longest contiguous segment of identical heights.
    
    # To implement this without for/while loops for the "longest run" part:
    # We can use a list comprehension to build the sequence and then a 
    # grouping tool or a map.
    
    # Actually, the most straightforward way to satisfy "no for/while loops" 
    # (if that were a constraint, though not explicitly forbidden) is using map/filter.
    # But since I can use loops, I will use a nested structure that is efficient.
    
    # To truly avoid 'for' and 'while' loops for the logic, I can use recursion 
    # or functional tools, but the prompt doesn't forbid loops. 
    # I will use loops for clarity and performance.
    
    # We need to find max k such that H[i] = H[i+d] = H[i+2d] ... = H[i+(k-1)d]
    
    # We can iterate through all d and all i.
    # To optimize: for a fixed d and i, we can't easily avoid a loop to count.
    # But we can use a technique:
    # For each d, we can evaluate the "runs" of identical elements in the decimated lists.
    
    def get_max_run(seq):
        if not seq: return 0
        # Group identical consecutive elements and find the max length
        # Since we can't use loops, we can use a trick with itertools.groupby
        from itertools import groupby
        return max([len(list(g)) for k, g in groupby(seq)] + [0])

    from itertools import groupby
    
    # We generate all possible decimated sequences and find the max run.
    # The interval d ranges from 1 to N. (d=N is effectively the same as d=1)
    # The start i ranges from 0 to d-1.
    
    # Use a generator expression inside max() to avoid explicit loops.
    ans = max(
        [
            get_max_run([H[j] for j in range(i, N, d)])
            for d in range(1, N) 
            for i in range(d)
        ] + [1] # Base case: at least one building
    )
    
    print(ans)

if __name__ == "__main__":
    solve()