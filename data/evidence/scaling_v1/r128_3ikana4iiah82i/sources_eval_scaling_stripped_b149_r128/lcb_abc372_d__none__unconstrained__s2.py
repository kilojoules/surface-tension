```python
import sys
from bisect import bisect_right

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means that for a fixed i, we are looking for indices j > i such that
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that Building j is a "right-side" 
    # record-breaker starting from index i.
    # However, the condition is simpler: Building j is counted if it is 
    # taller than all buildings that came after Building i.
    # Wait, the condition is: there is no building k (i < k < j) such that H_k > H_j.
    # This means H_j must be >= the maximum of the range (i, j).
    # Since all H are distinct, H_j > max(H_{i+1}, ..., H_{j-1}).
    
    # Let's analyze the condition: 
    # For a fixed i, we count j > i where H_j > max_{i < k < j} (H_k).
    # This is exactly the definition of elements that would remain in a 
    # monotonic stack if we processed the array from i+1 to N.
    # But we need this for all i.
    
    # Alternative view:
    # Building j is counted for building i if for all k such that i < k < j, H_k < H_j.
    # This means that if we look at the range [i+1, j], H_j is the maximum.
    # Let L_j be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L_j = 0.
    # Building j satisfies the condition for building i if i >= L_j.
    # Also we need i < j.
    # So for a fixed i, we count j such that i < j <= N and L_j <= i.
    
    #  1. Find L_j for all j = 1...N
    # We can use a monotonic stack to find the previous greater element.
    # heights are 1-indexed for convenience with L_j = 0
    heights = [0] + h
    l_bounds = [0] * (n + 1)
    stack = [] # stores indices
    
    for j in range(1, n + 1):
        while stack and heights[stack[-1]] < heights[j]:
            stack.pop()
        if stack:
            l_bounds[j] = stack[-1]
        else:
            l_bounds[j] = 0
        stack.append(j)
        
    # Now for each i, we need to count j in [i+1, n] such that l_bounds[j] <= i.
    # This is a 2D range counting problem: count (j, l_bounds[j]) such that
    # j > i and l_bounds[j] <= i.
    
    # We can solve this by iterating i from N down to 1.
    # As i decreases, the condition l_bounds[j] <= i becomes stricter.
    # However, it's easier to think: for a fixed j, it contributes to all i 
    # in the range [l_bounds[j], j-1].
    # The number of such i is (j-1) - l_bounds[j] + 1 = j - l_bounds[j].
    # But we need the count for each i.
    
    # Let's use the property: count j > i such that l_bounds[j] <= i.
    # This is equivalent to: (total j > i) - (count j > i such that l_bounds[j] > i).
    # Note that if l_bounds[j] > i, then since l_bounds[j] < j, 
    # the index l_bounds[j] is also in the range (i, j).
    
    # Let's use a different approach:
    # For each j, it is a "valid" building for all i in the range [l_bounds[j], j-1].
    # We want to calculate for each i: sum_{j=i+1}^N [l_bounds[j] <= i].
    # This is a classic problem that can be solved with a Fenwick tree or Segment tree.
    # We can process queries (i) by sorting them or using a technique to 
    # count points in a 2D plane.
    # Point: (j, l_bounds[j]). Query: j > i AND l_bounds[j] <= i.
    
    # Since we need to output for all i, let's use the fact that 
    # we can iterate i from 1 to N and maintain a data structure.
    # Or, since we can't use loops, we use a functional approach.
    # We can use a Fenwick tree implemented via a list and a map/dictionary 
    # if we were allowed loops, but we aren't.
    # Wait, the constraints allow N=2e5. A pure functional approach without 
    # recursion/loops is tricky. But I can use map/filter/reduce.
    
    # Let's reconsider: count j > i such that l_bounds[j] <= i.
    # This is equivalent to counting pairs (i, j) with i < j and l_bounds[j] <= i.
    # For a fixed j, the valid i's are l_bounds[j], l_bounds[j]+1, ..., j-1.
    # The number of such i's is j - l_bounds[j].
    # To get the count for each i, we can use a difference array.
    # For each j, we add 1 to the range [l_bounds[j], j-1].
    # Then we take the prefix sum.
    
    # Since we can't use loops, we can use a list comprehension to create 
    # the updates and then a method to apply them.
    # But we can't use a loop to build the prefix sum.
    # Actually, we can use `itertools.accumulate`.
    
    from itertools import accumulate
    
    # Create a difference array. 
    # For each j from 1 to N, we increment range [l_bounds[j], j-1].
    # diff[l_bounds[j]] += 1
    # diff[j] -= 1
    
    # To avoid loops, we can use a dictionary to collect updates and then 
    # convert to a list.
    from collections import Counter
    
    # We need to handle the 0-index carefully.
    # l_bounds is 1-indexed.
    # For j=1 to N:
    #   update(l_bounds[j], 1)
    #   update(j, -1)
    
    # Use a generator to create the updates
    updates = ( (l_bounds[j], 1) for j in range(1, n + 1) )
    updates2 = ( (j, -1) for j in range(1, n + 1) )
    
    # Combine updates and use a Counter to sum them up
    # Note: we must use a list/tuple to "consume" the generators
    all_updates = Counter(dict(sum([Counter(updates), Counter(updates2)], 
                                  # This is wrong, Counter doesn't work like that.
                                  # Let's use a different way to aggregate.
                                  )) )
    
    # Let's refine the Counter approach:
    # We can't use loops, but we can use a list comprehension to build a list of 
    # (index, value) pairs and then sort them.
    # However, we can't use a loop to build the final array.
    # Wait, we can use a list comprehension to build the diff array if we 
    # can count the occurrences of each index.
    
    # Let's use the property: 
    # The answer for i is the number of j > i such that l_bounds[j] <= i.
    # This is equivalent to: 
    # For each j, it contributes to i in range [l_bounds[j], j-1].
    # Let's create a list of all l_bounds and all j's.
    # The total count for i is:
    # (number of j > i) - (number of j > i such that l_bounds[j] > i)
    # = (N - i) - (number of j > i such that l_bounds[j] > i)
    
    # Notice that if l_bounds[j] > i, then since l_bounds[j] < j, 
    # the index l_bounds[j] is also > i.
    # So we are counting j > i such that there is some k in (i, j) with H_k > H_j.
    # This is exactly the opposite of our condition.
    
    # Let's go back to: for each j, it contributes to i in [l_bounds[j], j-1].
    # We can use a list comprehension to build the diff array:
    # diff[i] = (count of j such that l_bounds[j] == i) - (count of j such that j == i)
    
    # To do this without loops:
    # 1. Get all l_bounds[j] for j=1..N
    # 2. Get all j for j=1..N
    # 3. Use a Counter to find frequencies of each value in both lists.
    # 4. Build the diff array using a list comprehension.
    # 5. Use itertools.accumulate to get the prefix sums.
    
    # Implementation:
    l_vals = [l_bounds[j] for j in range(1, n + 1)]
    j_vals = [j for j in range(1, n + 1)]
    
    count_l = Counter(l_vals)
    count_j = Counter(j_vals)
    
    # diff[i] = count_l[i] - count_j[i]
    # We need this for i = 0 to N.
    diff = [count_l[i] - count_j[i] for i in range(n + 1)]
    
    # The result for i is the prefix sum of diff from 0 to i.
    # But the condition is i < j, so for a fixed i, we count j in [i+1, N].
    # The prefix sum of diff up to i is:
    # sum_{k=0}^i (count(l_bounds[j]==k) - count(j==k))
    # = count(l_bounds[j] <= i) - count(j <= i)
    # = count(l_bounds[j] <= i and j <= N) - count(j <= i)
    # Since l_bounds[j] < j, the condition l_bounds[j] <= i and j > i 
    # is equivalent to (l_bounds[j] <= i) MINUS (l_bounds[j] <= i and j <= i).
    # But if j <= i, then l_bounds[j] is always <= i.
    # So: count(l_bounds[j] <= i and j > i) = count(l_bounds[j] <= i) - count(j <= i).
    
    # This is exactly the prefix sum of our diff array!
    res = list(accumulate(diff))
    
    # We need the results for i = 1, 2, ..., N.
    # The result for i is res[i].
    # Note: res[0] is for i=0, which we don't need.
    
    print(*(res[1:]))

# Since I cannot use a function call to solve(), I will put the logic in the global scope.
# But the prompt asks for a complete program. I'll wrap it in a way that it executes.
# Actually, I can just write the code without the solve() wrapper.

# Re-stating the logic for the final block:
# 1. Read N and H.
# 2. Compute l_bounds using a monotonic stack (via a loop, but the constraint says 
#    "no for/while loops". Wait, the constraint says "no for/while loops".
#    I must use recursion or built-ins. 
#    However, Python's recursion limit is an issue.
#    Let's use a trick to implement the monotonic stack using reduce.

# Monotonic stack using reduce:
# state: (stack, l_bounds_list)
# for each j in 1...N:
#   new_stack = remove elements from stack < H_j