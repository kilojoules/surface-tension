The constraint to avoid explicit loops and recursion forces a functional programming style, which encourages the use of declarative transformations over imperative state changes. For this problem, I will use `sys.stdin.read().split()` to handle input, list comprehensions for data processing, and the `min`/`max` functions to determine the range of elements already in their correct positions. Specifically, I will identify the longest prefix and suffix of the permutation that are already sorted and contain the correct elements (i.e., $P_i = i$), and then determine if the remaining middle section can be solved in 0, 1, or 2 operations based on whether the elements in that section are simply a permutation of the required values.

```python
import sys

def solve():
    # Read all input at once and split into a flat list of strings
    data = sys.stdin.read().split()
    if not data:
        return
    
    # T is the first element
    T_str = data[0]
    T = int(T_str)
    
    # We process the data in chunks. Since we can't use loops, 
    # we use a list comprehension to iterate through the test cases.
    # We calculate the starting index for each test case based on N.
    # However, since N varies, we can't use a simple slice.
    # Instead, we'll use a helper to parse the flat list.
    
    # To handle variable N without loops, we can use a recursive-like 
    # structure via map/reduce or a custom iterator, but since 
    # recursion is banned, we use a generator/iterator 
    # and 'next()' inside a list comprehension.
    
    it = iter(data[1:])
    
    # For each test case:
    # 1. Get N
    # 2. Get P as a list of ints
    # 3. Find the first index i where P[i] != i+1
    # 4. Find the last index j where P[j] != j+1
    # 5. If no such i, answer is 0.
    # 6. If the elements between i and j are a permutation of (i+1...j+1),
    #    and we can pick a k such that sorting [0...k-1] and [k+1...N-1]
    #    fixes it, answer is 1.
    #    Actually, the operation is: pick k, sort P[0...k-2] and P[k...N-1].
    #    This means P[k-1] remains untouched.
    #    If there exists k such that P[k-1] == k and sorting the rest fixes it,
    #    then 1 operation suffices.
    #    Sorting the rest fixes it if the set of elements {P_0...P_{k-2}, P_k...P_{N-1}}
    #    is exactly {1...k-1, k+1...N}.
    #    This is true if and only if P[k-1] == k.
    #    So, if there is any k in [i+1, j+1] such that P[k-1] == k, 
    #    and the elements outside [i, j] are already correct, 
    #    we check if one operation at k works.
    #    Wait, the operation sorts everything EXCEPT index k-1.
    #    If we pick k such that P[k-1] is already k, then sorting the 
    #    prefix and suffix will definitely result in P_i = i for all i.
    #    So if there is any k in [1, N] such that P[k-1] == k, 
    #    and we only need to fix the range [i, j], 
    #    we can fix it in 1 op if k is in [i+1, j+1].
    #    Actually, if any P[k-1] == k exists for 1 <= k <= N, 
    #    we can potentially use it. But we need the elements 
    #    to the left of k to be {1...k-1} and right to be {k+1...N}.
    #    This is true if the "unsorted" part [i, j] is contained 
    #    within the ranges being sorted.
    #    If we pick k, we sort [0, k-2] and [k, N-1].
    #    This fixes everything if P[k-1] == k.
    
    # Correct Logic:
    # 0 ops: P is already (1...N)
    # 1 op: There exists k in [1, N] such that P[k-1] == k.
    # 2 ops: Otherwise.
    # Wait, Sample 3: (3, 2, 1, 7, 5, 6, 4). 
    # P[0]=3, P[1]=2, P[2]=1, P[3]=7, P[4]=5, P[5]=6, P[6]=4.
    # P[1]=2 is the only P[k-1]=k. If k=2, we sort P[0...0] and P[2...6].
    # P becomes (3, 2, 1, 4, 5, 6, 7). Not sorted.
    # The condition for 1 op is: there exists k such that P[k-1] == k 
    # AND sorting [0, k-2] and [k, N-1] results in (1...N).
    # This happens if {P_0...P_{k-2}} = {1...k-1} and {P_k...P_{N-1}} = {k+1...N}.
    # This is equivalent to saying the range [i, j] of elements where P_x != x+1
    # must not contain k-1, OR k-1 must be the only element in [i, j] that is correct.
    # Actually, simpler: 1 op is possible if there exists k such that 
    # P[k-1] == k AND (k-1 < i or k-1 > j). 
    # No, that's not right.
    # Let',s re-evaluate: 1 op with k works if P[k-1] == k AND 
    # max(P[0...k-2]) < k and min(P[k...N-1]) > k.
    
    # Let's use the property: 1 op is possible if there is some k 
    # such that P[k-1] == k and the set of elements is correct.
    # Since it's a permutation, P[k-1] == k implies the other N-1 
    # elements are a permutation of {1...k-1, k+1...N}.
    # Sorting them will always result in (1...k-1, k, k+1...N).
    # So 1 op is possible if and only if there exists k such that P[k-1] == k.
    # BUT, the operation is: sort 1 to k-1 and k+1 to N.
    # This is index 0 to k-2 and k to N-1.
    # This works if and only if P[k-1] is already k.
    # Sample 3: P = (3, 2, 1, 7, 5, 6, 4). P[1] = 2. k=2.
    # Sort P[0...0] and P[2...6]. P becomes (3, 2, 1, 4, 5, 6, 7).
    # Still not sorted because P[0] was 3 and it stayed 3.
    # Wait, the sample says Sample 3 takes 2 ops.
    # In Sample 3, P[1]=2, P[4]=5, P[5]=6.
    # For k=2: sort P[0...0] (3) and P[2...6] (1, 7, 5, 6, 4) -> (3, 2, 1, 4, 5, 6, 7)
    # For k=5: sort P[0...3] (3, 2, 1, 7) and P[5...6] (6, 4) -> (1, 2, 3, 7, 5, 4, 6)
    # For k=6: sort P[0...4] (3, 2, 1, 7, 5) and P[6...6] (4) -> (1, 2, 3, 5, 7, 6, 4)
    # None of these result in (1...7) in one go.
    # The condition for 1 op: there exists k such that P[k-1] == k 
    # AND sorting the others fixes it.
    # Sorting the others fixes it if the elements in the prefix are < k 
    # and elements in the suffix are > k.
    # That means for all x < k-1, P[x] < k and for all x > k-1, P[x] > k.
    # This is equivalent to saying that for the range [i, j] where P[x] != x+1,
    # the index k-1 must be outside this range [i, j].
    # Or more simply: 1 op is possible if there is some k such that 
    # P[k-1] == k and (k-1 < i or k-1 > j).
    # But if P is already sorted, 0 ops.
    # If P is not sorted, we need to find if any P[k-1] == k exists 
    # such that it's not "inside" the unsorted part.
    # Actually, if P[k-1] == k, then sorting the other two parts 
    # will result in a sorted array if and only if 
    # the set {P_0...P_{k-2}} is {1...k-1} and {P_k...P_{N-1}} is {k+1...N}.
    # This is true if and only if for all x < k-1, P[x] < k and for all x > k-1, P[x] > k.
    
    # Let's refine:
    # Let i be the first index where P[i] != i+1.
    # Let j be the last index where P[j] != j+1.
    # If no such i, 0 ops.
    # If there exists k such that P[k-1] == k and (k-1 < i or k-1 > j), 1 op.
    # Wait, if k-1 < i, then P[0...k-1] is already sorted. 
    # If we pick such a k, we sort P[0...k-2] (already sorted) 
    # and P[k...N-1]. Since P[0...i-1] is already sorted, 
    # sorting P[k...N-1] will fix everything if the only 
    # unsorted part was to the right of k.
    # This is possible if we can pick k such that P[k-1] == k 
    # and the unsorted part [i, j] is entirely contained in 
    # either [0, k-2] or [k, N-1].
    # This means k-1 > j or k-1 < i.
    
    # Let's check Sample 3: P = (3, 2, 1, 7, 5, 6, 4)
    # i = 0 (P[0]=3 != 1), j = 6 (P[6]=4 != 7)
    # Range [i, j] is [0, 6].
    # Is there any k-1 < 0 or k-1 > 6? No.
    # So 2 ops.
    # Sample 1: P = (2, 1, 3, 5, 4)
    # i = 0 (P[0]=2 != 1), j = 4 (P[4]=4 != 5)
    # Range [i, j] is [0, 4].
    # Wait, P[2] = 3. So k-1 = 2.
    # Is k-1 < i or k-1 > j? 2 < 0 or 2 > 4? No.
    # But Sample 1 says 1. Let me re-read.
    # "Choose k... sort 1 to k-1 and k+1 to N."
    # Sample 1: k=3. Sort P[0...1] (2, 1) and P[3...4] (5, 4).
    # P becomes (1, 2, 3, 4, 5).
    # This worked because P[k-1] was already k (P[2]=3), 
    # AND the unsorted parts were separated by k.
    # The unsorted parts are [