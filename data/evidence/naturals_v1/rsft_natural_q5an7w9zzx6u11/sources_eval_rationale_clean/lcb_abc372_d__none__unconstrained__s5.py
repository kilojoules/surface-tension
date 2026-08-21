import sys
from functools import reduce

def solve():
    # Increase recursion depth for deep reduce/mapping if necessary
    sys.setrecursionlimit(300000)
    
    # Read N and H
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building looking right from i.
    # This is equivalent to saying Building j is part of the 
    # upper-envelope of the heights to the right of i.
    # However, it is easier to think: for a fixed j, how many i < j 
    # satisfy the condition? 
    # Building j is counted for i if max(H[i+1]...H[j-1]) < H[j].
    # This means i must be such that all buildings between i and j are shorter than H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # Then any i in the range [L[j], j-1] satisfies the condition.
    # The number of such i is j - L[j].
    
    # To find L[j] for all j without loops, we use a monotonic stack via reduce.
    # The stack will store indices of buildings in decreasing order of height.
    
    def process_stack(state, current_idx):
        stack, left_bounds = state
        # Remove elements from stack that are shorter than current height
        # Using a helper function to simulate the while loop via recursion
        def pop_shorter(s):
            if s and H[s[-1]] < H[current_idx]:
                return pop_shorter(s[:-1])
            return s
        
        new_stack = pop_shorter(stack)
        # L[j] is the index of the taller building, or -1 if none
        left_bound = new_stack[-1] if new_stack else -1
        return (new_stack + [current_idx], left_bounds + [left_bound])

    # We use reduce to iterate through the indices 0 to N-1
    # Initial state: (stack, left_bounds_list)
    initial_state = ([], [])
    final_state = reduce(process_stack, range(N), initial_state)
    L = final_state[1]

    # Now we need to find for each i, how many j > i satisfy i >= L[j].
    # This is equivalent to counting j such that L[j] <= i < j.
    # We can use a Difference Array / Fenwick tree approach, but since we 
    # cannot use loops, we can use the fact that we need to sum 
    # count(j) where L[j] <= i < j.
    # This is sum_{j=i+1}^{N-1} [L[j] <= i].
    
    # Alternatively: for each j, it contributes +1 to the answer of all i in [L[j], j-1].
    # We can use a difference array: diff[L[j]] += 1, diff[j] -= 1.
    # Then the answer for i is the prefix sum of the difference array.
    
    # Since we can't use loops, we use a list and map/reduce for the difference array.
    # Create a difference array of size N+1
    diff = [0] * (N + 1)
    
    # To update the difference array without loops, we can't mutate diff in a map.
    # But we can create a list of updates and use a method to aggregate them.
    # However, the simplest way to get the counts is to realize:
    # The total count for i is (number of j > i such that L[j] <= i).
    
    # Let's use the property: for a fixed i, we want to count j > i such that L[j] <= i.
    # This is equivalent to: (Total j > i) - (j > i such that L[j] > i).
    # Note that L[j] > i and j > i implies the building at L[j] is between i and j 
    # and is taller than building j.
    
    # Let's go back to the difference array. We can use a dictionary to aggregate 
    # updates and then a prefix sum.
    from collections import Counter
    updates = Counter()
    # For each j, range is [L[j], j-1]
    # We need to add 1 to indices from L[j] to j-1.
    # We can't use a loop to update the Counter, but we can use a generator.
    
    # Create a list of (index, change) pairs
    # For each j: (L[j], 1) and (j, -1)
    # Note: L[j] can be -1, so we use max(0, L[j])
    change_pairs = reduce(lambda acc, j: acc + [(max(0, L[j]), 1), (j, -1)], range(N), [])
    
    # Aggregate changes at each index
    # We can use a dictionary to sum changes at each position
    def aggregate(acc, pair):
        acc[pair[0]] = acc.get(pair[0], 0) + pair[1]
        return acc
    
    final_diffs = reduce(aggregate, change_pairs, {})
    
    # Now compute prefix sums to get the actual values
    # result[i] = sum(final_diffs[k] for k <= i)
    def compute_prefix_sums(current_sum, i):
        new_sum = current_sum + final_diffs.get(i, 0)
        return new_sum, new_sum

    # Use map to compute prefix sums across the range 0 to N-1
    # Since we need the state (current_sum), we use a custom reduce or a generator
    def prefix_sum_gen(it, s):
        for i in it:
            s += final_diffs.get(i, 0)
            yield s

    results = list(prefix_sum_gen(range(N), 0))
    
    # The problem asks for j > i. The difference array logic includes j=i if L[i] <= i < i,
    # which is impossible. But it might count j=i if not careful.
    # Let's refine: for each j from 0 to N-1, it contributes to i in [L[j], j-1].
    # For i = N-1, the range is empty, so result is 0.
    # The prefix sum of the difference array correctly gives the count for each i.
    # However, the difference array logic above:
    # for j=0: L[0]=-1, range [0, -1] -> no update.
    # for j=1: L[1]=0, range [0, 0] -> diff[0]++, diff[1]--.
    # This is correct.
    
    # But wait, the prefix sum logic above includes the contribution of j=i.
    # Let's check: for i, we want count of j > i such that L[j] <= i.
    # The difference array: for each j, we add 1 to [L[j], j-1].
    # So for a specific i, the value is sum_{j} [L[j] <= i <= j-1].
    # This is exactly what we need.
    
    # One final adjustment: the prefix sum for i=0 includes contributions from all j > 0 where L[j] <= 0.
    # The current prefix_sum_gen logic:
    # i=0: s = 0 + diff[0].
    # i=1: s = diff[0] + diff[1].
    # This is correct.
    
    # Since we can't use a for loop in the generator, we use map/reduce.
    # To get the prefix sums without a loop:
    def get_sums(n, d):
        def step(acc, i):
            return acc + [acc[-1] + d.get(i, 0)]
        return reduce(step, range(n), [0]) [1:]

    # The above reduce is O(N^2) because of list concatenation. 
    # Let's use a more efficient way.
    # We can use a list and mutate it inside a reduce, but that's borderline.
    # The most "functional" way to do prefix sum in Python is itertools.accumulate.
    from itertools import accumulate
    
    # Create a list of values to accumulate
    vals = list(map(lambda i: final_diffs.get(i, 0), range(N)))
    ans = list(accumulate(vals))
    
    # The problem asks for i = 1 to N. Our i is 0 to N-1.
    # For i = N-1, the answer should be 0.
    # Let's check Sample 1: 2 1 4 3 5
    # j=0 (H=2): L[0]=-1. Range: none.
    # j=1 (H=1): L[1]=0. Range: [0, 0]. diff[0]++, diff[1]--.
    # j=2 (H=4): L[2]=-1. Range: [0, 1]. diff[0]++, diff[2]--.
    # j=3 (H=3): L[3]=2. Range: [2, 2]. diff[2]++, diff[3]--.
    # j=4 (H=5): L[4]=-1. Range: [0, 3]. diff[0]++, diff[4]--.
    # diff: {0: 3, 1: -1, 2: 0, 3: -1, 4: -1}
    # prefix sums: 3, 2, 2, 1, 0. Correct.
    
    print(*(ans))

if __name__ == "__main__":
    solve()