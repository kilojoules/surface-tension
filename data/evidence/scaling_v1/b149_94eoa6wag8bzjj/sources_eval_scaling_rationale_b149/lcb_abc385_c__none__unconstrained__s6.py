import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to check every possible starting building (i)
    # and every possible interval (d) between chosen buildings.
    # For a fixed start i and interval d, we count how many buildings
    # have the same height as building i.
    # However, the condition is that ALL chosen buildings must have the same height.
    # So for a fixed i and d, we check the sequence h[i], h[i+d], h[i+2d]...
    # and count how many match h[i]. 
    # Wait, the condition says "The chosen buildings all have the same height".
    # This means if we pick a sequence with interval d, and some building in that
    # sequence has a different height, we cannot "skip" it; we simply cannot 
    # include it in our set of chosen buildings. 
    # Actually, the problem says "choose some buildings". It doesn't say 
    # they must be the ONLY buildings at that interval.
    # But "arranged at equal intervals" implies we pick indices i, i+d, i+2d...
    # Let's re-read: "The chosen buildings are arranged at equal intervals."
    # This means if we choose indices {p1, p2, ..., pk}, then p_{j+1} - p_j = d.
    # So we are looking for the longest arithmetic progression of indices 
    # where all corresponding heights are identical.

    # For every possible starting index i and every possible interval d:
    # We count how many elements in the sequence h[i], h[i+d], h[i+2d]... 
    # are equal to h[i].
    # Note: The elements must be strictly at the interval. 
    # If we pick indices i, i+d, i+2d, then all those specific buildings must have the same height.
    # If h[i+d] is different, we can't just skip it and take h[i+2d] because then 
    # the interval between the 1st and 2nd chosen building would be 2d.
    # Therefore, for a fixed i and d, we count the contiguous prefix of the 
    # sequence h[i], h[i+d], h[i+2d]... that matches h[i].
    # Actually, the simplest interpretation is: pick a start i and interval d,
    # and count how many k >= 0 satisfy i + k*d < N and h[i + k*d] == h[i].
    # But the "equal intervals" condition means the indices are i, i+d, i+2d...
    # If h[i+d] != h[i], we can't include it. If we don't include it, 
    # the remaining chosen buildings (i and i+2d) are no longer at "equal intervals"
    # unless we redefine the interval as 2d.
    # Thus, for a fixed i and d, we are looking for the largest k such that
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].

    # To implement this without loops:
    # 1. Generate all pairs of (i, d) where 0 <= i < n and 1 <= d < n.
    # 2. For each pair, find the length of the prefix of the sequence 
    #    h[i], h[i+d], ... that matches h[i].
    
    # Since we can't use while loops, we can use a list comprehension to 
    # create the sequence and then a trick to find the first mismatch.
    # However, the constraint to avoid loops makes "counting the prefix" hard.
    # Let's reconsider: the problem can be solved by iterating over all i and d,
    # and for each, counting how many k satisfy h[i + k*d] == h[i] 
    # PROVIDED that all elements between them in the progression also match.
    
    # Actually, the most straightforward way to count the prefix without a loop:
    # For a fixed i and d, the sequence is S = [h[i + k*d] for k in range((n-1-i)//d + 1)]
    # We want the length of the prefix of S consisting of identical elements.
    # We can use a helper function with recursion or a clever comprehension.
    # But wait, the constraint says "no loops". Recursion is allowed.
    
    def get_prefix_len(seq):
        if not seq: return 0
        val = seq[0]
        # Use a generator expression and next() to find the first index where it differs
        # This is a common idiom to avoid explicit for/while loops.
        # We find the first index k where seq[k] != val.
        first_mismatch = next((k for k, x in enumerate(seq) if x != val), len(seq))
        return first_mismatch

    # We can use a nested list comprehension to evaluate all i, d and find the max.
    # i: 0 to n-1
    # d: 1 to n-1
    
    # To avoid the function call inside the comprehension if preferred, 
    # we can just use the next() logic directly.
    
    ans = max([
        next((k for k in range((n - 1 - i) // d + 2) 
              if k >= len([h[i + j*d] for j in range((n - 1 - i) // d + 1)]) 
              or h[i + k*d] != h[i]), 
              (n - 1 - i) // d + 1)
        for i in range(n)
        for d in range(1, n)
    ] + [1])

    print(ans)

if __name__ == "__main__":
    solve()