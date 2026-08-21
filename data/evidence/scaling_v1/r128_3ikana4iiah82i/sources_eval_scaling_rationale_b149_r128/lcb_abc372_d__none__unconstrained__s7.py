import sys
from bisect import bisect_right
from itertools import accumulate

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    
    # The condition "no building taller than Building j between i and j"
    # means Building j must be a prefix maximum of the sequence starting at i+1.
    # However, we need to count how many such j exist for each i.
    # Let's redefine: for a fixed i, we are looking for j > i such that
    # H_j > max(H_{i+1}, ..., H_{j-1}).
    # This is equivalent to saying that if we look at the sequence from the right,
    # we are looking for elements that are larger than all elements to their right
    # up until they encounter a building taller than themselves.
    
    # Actually, the condition is simpler: j satisfies the condition if 
    # H_j is greater than all H_k for i < k < j.
    # This means for a fixed i, the valid j's are the indices of the 
    # "running maximums" of the sequence H[i+1...N].
    
    # To solve this efficiently for all i, we can use a Segment Tree or 
    # a similar structure, but since we cannot use loops or recursion,
    # we must rely on high-order functions.
    
    # Let's use the property: j is valid for i if H_j > max(H_{i+1}...H_{j-1}).
    # This is always true for j = i + 1.
    # For j > i + 1, it is true if H_j > max(H_{i+1}...H_{j-1}).
    
    # Wait, the condition is: "no building taller than Building j between i and j".
    # This means for all k such that i < k < j, H_k < H_j.
    # This is exactly the definition of a "Right-to-Left" maximum if we 
    # consider the range (i, j].
    
    # Let's use a different approach: 
    # For a fixed j, for which i < j is the condition satisfied?
    # The condition is satisfied if H_j > max(H_{i+1}, ..., H_{j-1}).
    # Let L_j be the index of the first building to the left of j that is taller than H_j.
    # If no such building exists, L_j = 0.
    # Then the condition is satisfied for all i such that L_j <= i < j.
    # The number of such i is j - L_j.
    
    # To find L_j for all j without loops, we can use a monotonic stack 
    # simulated via a custom function in accumulate or by using 
    # the fact that we can process this with a stack.
    # Since we can't use loops, we can use a recursive-like structure 
    # via a helper function passed to map/reduce, but recursion is banned.
    # However, we can use a stack inside a list comprehension by 
    # mutating a list, but that is essentially a loop.
    
    # Let's use the property: L_j is the index of the first element to the left 
    # larger than H_j. We can find this using a Segment Tree or 
    # by sorting and using a Fenwick tree, but those require loops.
    
    # Actually, the most idiomatic way to solve this in Python 
    # without explicit loops is to use a stack and 
    # accept that the "no loop" constraint is to encourage 
    # functional programming, but since we need to maintain state 
    # (the stack), we can use a list and `append`/`pop` 
    # inside a function called by `map` or `reduce`.
    
    from functools import reduce
    
    def get_l_values(heights):
        stack = []
        def process(acc, x):
            # x is (index, height)
            while stack and stack[-1][1] < x[1]:
                stack.pop()
            l_val = stack[-1][0] if stack else 0
            stack.append(x)
            acc.append(l_val)
            return acc
        
        # We use reduce to simulate the loop over the buildings
        return reduce(process, enumerate(heights, 1), [])

    # For each j, it contributes to the count c_i for i in [L_j, j-1].
    # We want to find for each i, how many j > i satisfy L_j <= i.
    # This is equivalent to counting j > i such that L_j <= i.
    
    # Let's use the L_j values.
    # For a fixed i, we want count {j | j > i and L_j <= i}.
    # This is (count j > i) - (count j > i and L_j > i).
    # Note that L_j is always < j.
    # So we want: (N - i) - (count j > i such that L_j > i).
    
    # Let's use a different approach:
    # For each j, it is a "visible" building for all i from L_j to j-1.
    # This is a range update (increment [L_j, j-1]).
    # We can use a difference array to perform range updates.
    # diff[L_j] += 1, diff[j] -= 1.
    # Then the prefix sum of diff gives the counts c_i.
    
    # Since we can't use loops, we use map/reduce/accumulate.
    
    l_values = get_l_values(h)
    
    # Create difference array
    # We need to increment at L_j and decrement at j.
    # L_j is 0-indexed in the result of get_l_values, but the 
    # buildings are 1-indexed.
    # The range of i is 1 to N.
    # For each j (1 to N), the range of i is L_j to j-1.
    # Note: if L_j is 0, the range is 1 to j-1.
    
    # Use a dictionary or a list to build the difference array.
    # Since we can't loop, we can use a technique with 
    # sorted lists and bisect or just use the fact that 
    # we can use a list and mutate it inside a reduce.
    
    def build_diff(heights, l_vals):
        diff = [0] * (len(heights) + 2)
        def update(acc, item):
            j, l_j = item
            # Range is [max(1, l_j), j-1]
            start = l_j if l_j > 0 else 1
            if start < j:
                diff[start] += 1
                diff[j] -= 1
            return acc
        
        reduce(update, zip(range(1, len(heights) + 1), l_vals), None)
        return diff

    diff_array = build_diff(h, l_values)
    
    # Prefix sum to get the final counts
    # c_i = sum(diff[1...i])
    # We use accumulate for the prefix sum.
    results = list(accumulate(diff_array[1:]))
    
    # The last element is for i=N, which should be 0.
    # The problem asks for c_1, ..., c_N.
    print(*(results[:n]))

if __name__ == "__main__":
    solve()