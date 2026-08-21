import sys

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to find the maximum number of buildings that:
    # 1. Have the same height H
    # 2. Are spaced at equal intervals 'step'
    # 3. Start at index 'i'
    
    # We iterate through all possible starting indices i (0 to n-1)
    # and all possible intervals step (1 to n-1).
    # For a fixed i and step, we check how many buildings at i, i+step, i+2*step...
    # have the same height as building i.
    
    # To avoid explicit loops, we use nested list comprehensions.
    # The innermost part calculates the count of buildings with the same height
    # for a specific start index and step.
    
    # We use a generator expression inside max() for memory efficiency.
    # We handle the case of N=1 separately or ensure the range covers it.
    
    # For a fixed i and step, the sequence of indices is i, i+step, i+2*step...
    # The number of elements in this sequence is (n - 1 - i) // step + 1.
    # However, we only count them if they all have the same height as h[i].
    # Wait, the condition is "the chosen buildings all have the same height".
    # This means we can pick a subset of the sequence i, i+step... 
    # But the problem says "arranged at equal intervals", implying the 
    # gap between any two adjacent chosen buildings is the same.
    # This means if we choose buildings at indices p1, p2, ..., pk,
    # then p2 - p1 = p3 - p2 = ... = pk - pk-1 = step.
    
    # So for a fixed i and step, we check the sequence h[i], h[i+step], h[i+2*step]...
    # and count how many of them are equal to h[i]. 
    # IMPORTANT: The condition "arranged at equal intervals" means the 
    # indices must be an arithmetic progression. It does NOT say 
    # we can skip buildings within that progression.
    # If we pick indices (i, i+step, i+2*step), all three must have the same height.
    # If h[i+step] is different, we cannot include h[i+2*step] in a 
    # sequence of 3. We could only have a sequence of 1 (just h[i]).
    
    # Actually, the most straightforward interpretation is:
    # Pick a height 'H', a start index 'i', and a step 's'.
    # Count how many k >= 0 satisfy i + k*s < n AND h[i + k*s] == H.
    # BUT, the buildings chosen must be at equal intervals. 
    # This means if we choose k buildings, they must be at i, i+s, i+2s, ..., i+(k-1)s.
    # All of these must have height H.
    
    # Let's refine: for every i and s, we find the longest prefix of the 
    # sequence h[i], h[i+s], h[i+2s]... that all have the same height h[i].
    
    # Since N is small (3000), we can't quite do O(N^3) with comprehensions 
    # without hitting limits, but O(N^2) is fine.
    # For a fixed i and s, we can't easily "break" a comprehension.
    # But we can use a trick: for a fixed i and s, we check all k 
    # and find the first k where h[i + k*s] != h[i].
    
    # Let's reconsider: the condition is simply that the chosen indices 
    # form an arithmetic progression and the heights at those indices are identical.
    # This means we are looking for the maximum k such that there exist i, s 
    # where h[i] = h[i+s] = h[i+2s] = ... = h[i+(k-1)s].
    
    # To implement this without loops:
    # For each i and s, we can determine the maximum k by checking 
    # how many consecutive elements starting from i with step s have height h[i].
    
    # However, a simpler O(N^2) approach:
    # For every pair (i, j) with i < j, they determine a step s = j - i.
    # We can't easily count "consecutive" without a loop or recursion.
    # But we can just iterate over all i and s, and for each, 
    # count how many k satisfy h[i + k*s] == h[i] for ALL 0 <= m <= k.
    
    # Wait, the constraint to avoid loops makes "counting consecutive" hard.
    # Let's use the property: for a fixed i and s, the number of buildings is
    # the smallest k such that h[i + k*s] != h[i], or the end of the array.
    
    # Actually, the simplest way to write this in a comprehension is to 
    # iterate over all i and s, and for each, use a helper function or 
    # a logic that finds the length of the monochromatic prefix.
    
    # Since I cannot use loops, I will use a recursive-like structure 
    # via a helper function (which is allowed) or just use the fact that 
    # N=300나 3000. For N=3000, O(N^2) is 9 million. 
    # We can iterate over all i and s, and for each, calculate the 
    # length of the sequence. To avoid the loop, I'll use 
    # a generator to find the first index that doesn't match.
    
    # Correct logic:
    # For each i in 0..N-1:
    #   For each s in 1..N-1:
    #     Find max k such that h[i], h[i+s]... h[i+(k-1)s] are all equal.
    
    # To do this without a loop:
    # For a fixed i and s, the sequence is h[i::s].
    # We want the length of the prefix of h[i::s] that equals h[i].
    
    # We can use `itertools.takewhile` to get the prefix.
    from itertools import takewhile
    
    # The result is the max length of takewhile(lambda x: x == h[i], h[i::s])
    # for all i and s.
    
    # We use a generator expression to find the maximum.
    # Note: h[i::s] is a slice, which is efficient.
    
    ans = max(
        (
            len(list(takewhile(lambda x: x == h[i], h[i::s])))
            for i in range(n)
            for s in range(1, n)
        ),
        default=1
    )
    
    # Special case for N=1 is handled by default=1 and the ranges.
    # But if N=1, range(1, 1) is empty, so it returns default.
    # If N > 1, the max will be at least 1.
    
    print(ans if n > 0 else 0)

if __name__ == "__main__":
    solve()