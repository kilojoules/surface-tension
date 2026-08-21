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
    # 2. Are spaced at equal intervals 'd'
    
    # For every possible starting index 'i' (0 to n-1)
    # and every possible interval 'd' (1 to n-1)
    # we check how many buildings starting from i with step d have the same height as h[i].
    
    # We use a nested list comprehension to evaluate all combinations of i and d.
    # For a fixed i and d, the number of buildings is the length of the 
    # longest prefix of the sequence h[i], h[i+d], h[i+2d]... that equals h[i].
    # However, the condition is that ALL chosen buildings must have the same height.
    # This means we are looking for the count of indices k such that h[i + k*d] == h[i].
    # Wait, the condition "arranged at equal intervals" implies we pick a subset 
    # {i, i+d, i+2d, ..., i+(m-1)d}. All these must have the same height.
    
    # Let's redefine: for every pair (i, d), we find the maximum m such that
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(m-1)d].
    
    # Since we cannot use loops, we can use a helper function with recursion 
    # or a clever comprehension. But recursion might hit depth limits.
    # Actually, the simplest way is:
    # For every i and d, we check the sequence and count how many consecutive 
    # elements match h[i].
    
    # To avoid loops and recursion, we can pre-calculate the counts.
    # For a fixed i and d, the number of buildings is:
    # m = 1 + (number of k > 0 such that h[i+k*d] == h[i] AND all elements before it also matched)
    
    # Alternatively, since N is small (3000), we can iterate over all i and d,
    # and for each, use a generator to find the first index that doesn't match.
    # But we can't use a loop to find that index.
    
    # Let's use the property: for a fixed i and d, we want the largest m 
    # such that for all 0 <= k < m, h[i + k*d] == h[i].
    # This is equivalent to finding the smallest k such that h[i + k*d] != h[i],
    # and the answer for that (i, d) is k.
    
    # We can use a list comprehension to create a boolean list for the sequence
    # and then find the first False.
    
    # To keep it strictly loop-free and efficient:
    # We can iterate over all possible heights present in the array.
    # For each height, find all indices where it occurs.
    # Then for every pair of indices (idx1, idx2), calculate the interval d = idx2 - idx1.
    # Then check how many subsequent indices (idx2 + d, idx2 + 2d...) also have that height.
    
    # Actually, the simplest loop-free approach is:
    # For every i in 0..N-1 and every d in 1..N-1:
    # Count how many k >= 0 satisfy h[i + k*d] == h[i] 
    # STOPPING at the first k where h[i + k*d] != h[i].
    
    # Since we can't use 'while', we can use a trick with `itertools.takewhile`.
    from itertools import takewhile
    
    # We generate all possible (i, d) pairs.
    # For each, we create a generator that yields indices i, i+d, i+2d...
    # We use takewhile to keep elements as long as h[index] == h[i].
    
    results = [
        sum(1 for _ in takewhile(lambda idx: idx < n and h[idx] == h[i], range(i, n, d)))
        for i in range(n)
        for d in range(1, n)
    ]
    
    # The answer is the max of results, or 1 if N=1.
    print(max(results) if results else 1)

if __name__ == "__main__":
    solve()