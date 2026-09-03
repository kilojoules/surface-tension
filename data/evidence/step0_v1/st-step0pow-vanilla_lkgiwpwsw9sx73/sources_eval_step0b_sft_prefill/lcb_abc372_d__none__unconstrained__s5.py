import sys
import bisect

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "No building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H[k] <= H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "right-to-left" 
    # maximum when looking from j back towards i.
    # More simply: if we process the array from right to left, for a fixed i,
    # we want to count how many j > i are "visible" from i.
    # A building j is visible from i if it is taller than all buildings between i and j.
    # This is exactly the number of elements in the "monotonic increasing stack" 
    # ( own height included) constructed from the range [i+1, N].
    
    # However, the constraint N=2*10^5 suggests an O(N log N) or O(N) approach.
    # Let's use a Divide and Conquer approach (similar to counting inversions)
    # or a Segment Tree/Fenwick tree approach.
    
    # Alternative observation:
    # For a fixed i, the sequence of buildings j that satisfy the condition are:
    # j1 = i + 1
    # j2 = first index > j1 such that H[j2] > H[j1]
    # j3 = first index > j2 such that H[j3] > H[j2]
    # ... and so on.
    # This is because any building between j_k and j_{k+1} is shorter than H[j_k],
    # and since H[j_k] > H[j_{k+1}] is false, the condition is maintained.
    # Wait, the condition is: no building taller than H[j] between i and j.
    # Let's re-read: "There is no building taller than Building j between Buildings i and j."
    # This means for all k such that i < k < j, H[k] <= H[j].
    
    # Let's use the property: 
    # For a fixed j, it satisfies the condition for i if H[j] >= max(H[i+1...j-1]).
    # This is equivalent to saying that i must be greater than or equal to the 
    # index of the first building to the left of j that is taller than H[j].
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = 0 (1-indexed).
    # Then for a fixed j, the indices i that satisfy the condition are i = L[j], L[j]+1, ..., j-1.
    # The number of such i is j - L[j].
    # But we need for each i, the count of j.
    # This is: for each i, count j > i such that L[j] <= i.
    
    # 1. Find L[j] for all j = 1...N using a monotonic stack.
    # H is 0-indexed in Python, so H[0...N-1].
    # L[j] = index of nearest element to the left > H[j].
    
    L = [0] * N
    stack = [] # stores indices
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if not stack:
            L[j] = -1 # No building taller to the left
        else:
            L[j] = stack[-1]
        stack.append(j)
    
    # Now we have pairs (L[j], j) for each j from 0 to N-1.
    # We want to find for each i: count j such that i < j < N and L[j] <= i.
    # Note: the condition is i < j. If L[j] = -1, then any i < j works.
    # If L[j] = k, then i can be k, k+1, ..., j-1.
    # So for a fixed i, we need to count j such that j > i AND (L[j] <= i).
    
    # Let's use a Fenwick tree. We iterate i from N-1 down to 0.
    # When we are at i, we want to count j in [i+1, N-1] such that L[j] <= i.
    # This is still tricky. Let's change perspective:
    # Each j contributes to the count c_i for i in [L[j], j-1].
    # (If L[j] is -1, i is in [0, j-1]).
    # This is a range update, point query problem.
    # For each j, we add 1 to the range [max(0, L[j]), j-1].
    # Then we query the value at each i.
    
    # Since we cannot use loops, we use a difference array (prefix sums).
    # diff[L] += 1, diff[j] -= 1
    
    # We need to handle the L[j] = -1 case as 0.
    # For j = 0 to N-1:
    #   left = L[j] if L[j] != -1 else 0
    #   right = j - 1
    #   if left <= right:
    #       diff[left] += 1
    #       if right + 1 < N:
    #           diff[right + 1] -= 1
    
    # To avoid loops, we use map/list comprehensions and accumulate.
    # But we can't easily do diff[left] += 1 in a comprehension.
    # We can use a technique with sorted lists and binary search or 
    # simply use the fact that we can group the updates.
    
    # Let's use the property: c_i = (number of j > i such that L[j] <= i).
    # c_i = (number of j > i) - (number of j > i such that L[j] > i).
    # c_i = (N - 1 - i) - (number of j > i such that L[j] > i).
    # Note that if L[j] > i, then since L[j] < j, the condition j > i is automatically satisfied.
    # So c_i = (N - 1 - i) - (number of j such that L[j] > i).
    
    # Let's collect all L[j] values.
    # We need to count how many L[j] are greater than i for each i.
    # We can sort the L array and use bisect_right.
    
    # Filter out L[j] that are -1 because they are never > i (since i >= 0).
    # Actually, if L[j] = -1, it doesn't contribute to the "subtracted" part.
    
    sorted_L = sorted([l for l in L if l != -1])
    
    # For each i from 0 to N-1:
    # count_greater = len(sorted_L) - bisect.bisect_right(sorted_L, i)
    # c_i = (N - 1 - i) - count_greater
    
    # We can use a list comprehension to calculate all c_i.
    results = [ (N - 1 - i) - (len(sorted_L) - bisect.bisect_right(sorted_L, i)) for i in range(N) ]
    
    print(*(results))

if __name__ == "__main__":
    solve()