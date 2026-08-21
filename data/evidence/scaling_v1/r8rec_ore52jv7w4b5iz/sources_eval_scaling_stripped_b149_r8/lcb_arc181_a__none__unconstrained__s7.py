```python
import sys
from itertools import groupby

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to process test cases without explicit for-loops
    # We use a helper function to extract the slices of the input list
    def get_cases(data, t):
        # This is a trick to avoid loops: we map a function over a range
        # but since we can't use loops, we calculate offsets.
        # However, N varies per case, so we must process sequentially.
        # To strictly avoid 'for' and 'while', we use a recursive-like 
        # structure via a list comprehension or map, but N is variable.
        # Actually, the constraint says "no for/while". 
        # We can use a generator that yields the result for each case.
        pass

    # Since N is variable, we can't pre-calculate offsets easily.
    # But we can use a trick: process the flat list using a custom reducer 
    # or a recursive-like structure. 
    # Wait, the most reliable way to handle variable N without loops 
    # is to use a generator that consumes the input.
    
    def process_all(data):
        # We use a generator to maintain state of the current index
        # But we can't use a loop to iterate the generator.
        # We can use a recursive function, but Python has a recursion limit.
        # Alternatively, we can use a trick with a list and a mutable state object.
        state = {'idx': 1}
        
        def handle_case(_):
            n = int(data[state['idx']])
            p = data[state['idx'] + 1 : state['idx'] + 1 + n]
            state['idx'] += 1 + n
            
            # The core logic:
            # 0 ops: already sorted
            # 1 op: there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
            # This happens if there is some k where P[k] is the only element 
            # that could be "out of place" relative to the sorted version,
            # OR more simply: if we can pick k such that all elements 
            # {P_1...P_{k-1}} are {1...k-1} and {P_{k+1}...P_N} are {k+1...N}.
            # Actually, the condition for 1 op is:
            # There exists k such that sorting the two partitions results in (1...N).
            # This is true if and only if the elements that are NOT in their 
            # correct sorted positions form a contiguous block that can be 
            # "split" by a single k.
            # More simply: 1 op is possible if there is a k such that 
            # the set of values {P_1...P_{k-1}} is {1...k-1} 
            # AND the set of values {P_{k+1}...P_N} is {k+1...N}.
            # This is equivalent to saying P_k = k and the remaining 
            # elements are partitioned correctly.
            # Actually, the simplest condition for 1 op:
            # There exists k such that sorting [0, k-1) and [k+1, N) 
            # results in [1, 2, ..., N].
            # This happens if and only if:
            # For all i < k, P_i is in the set {1, ..., k-1} (not necessarily sorted)
            # AND for all i > k, P_i is in the set {k+1, ..., N}.
            # This implies P_k must be k.
            # And the elements to the left are a permutation of 1...k-1,
            # and elements to the right are a permutation of k+1...N.
            
            # Let's refine:
            # 0 ops: P is already sorted.
            # 1 op: There exists k such that P_k = k, and 
            # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
            # 2 ops: Always possible for N >= 3.
            
            # To check the 1-op condition without loops:
            # We need to find if any k satisfies:
            # (prefix_max[k-1] < k) and (suffix_min[k+1] > k) and (P[k] == k)
            # We can use list comprehensions and built-ins.
            
            # Since we can't use loops, we use map/filter/reduce.
            # But we need prefix/suffix arrays. We can't use reduce for 
            # prefix/suffix without a loop. 
            # Wait, we can use a trick: 
            # A permutation is sorted if it equals sorted(P).
            # 1 op is possible if there is a k such that:
            # sorted(P[0:k]) + [P[k]] + sorted(P[k+1:]) == [1, 2, ..., N]
            # This is true if P[k] == k+1 and max(P[0:k]) < k+1 and min(P[k+1:]) > k+1.
            
            # To avoid loops for prefix/suffix, we can use the fact that
            # P_i = i for all i is the goal.
            # The only way 1 op works is if there is some k where 
            # the elements {P_1...P_{k-1}} are exactly {1...k-1} 
            # and {P_{k+1}...P_N} are exactly {k+1...N}.
            # This means for that k, P_k = k, and for all i < k, P_i < k, 
            # and for all i > k, P_i > k.
            
            # This is equivalent to: 
            # There exists k such that prefix_max[k-1] == k-1 and suffix_min[k+1] == k+1.
            # We can compute prefix_max and suffix_min using a trick with 
            # a list and a function that updates it, but that's essentially a loop.
            # However, we can use the property:
            # 1 op is possible if there is at least one k such that 
            # P[k] == k+1 and the number of elements P[i] < k+1 for i < k 
            # is exactly k.
            
            # Let's use the property: 1 op is possible if there is a k such that
            # the set of indices i where P[i] != i+1 is contained within 
            # [0, k-1] UNION [k+1, N-1], and the elements in those 
            # ranges are just permutations of the correct values.
            # Actually, the simplest condition:
            # 1 op is possible if there exists k such that:
            # sorted(P[:k]) == [1...k] AND sorted(P[k+1:]) == [k+2...N]
            # (assuming 1-indexing for values).
            # This is true if max(P[:k]) == k and min(P[k+1:]) == k+2.
            
            # To implement this without loops:
            # We can use a list comprehension to check all k.
            # But max()/min() inside a list comprehension is O(N^2).
            # We need O(N). 
            # We can use a technique to get prefix/suffix without loops:
            # In Python 3.8+, we can't use loops, but we can use 
            # a recursive-like structure via a list and a helper.
            # Actually, the most "legal" way to do prefix/suffix without 
            # for/while is using a custom function with a list 
            # and calling it via map/list comprehension.
            
            # Let's use a different approach:
            # 1 op is possible if there is some k such that 
            # P[k] == k+1 and for all i, (P[i] < k+1 iff i < k).
            # This is equivalent to: 
            # the set of indices {i | P[i] != i+1} does not include k,
            # and for all i in that set, if i < k then P[i] < k+1, 
            # and if i > k then P[i] > k+1.
            
            # Let S be the set of indices where P[i] != i+1.
            # If S is empty, 0 ops.
            # If S is not empty, 1 op is possible if there exists k 
            # such that k is not in S, and for all i in S:
            # (i < k and P[i] < k+1) or (i > k and P[i] > k+1).
            # This is equivalent to:
            # max(i for i in S if i < k) < k < min(i for i in S if i > k)
            # is NOT the condition.
            # The condition is: all i in S that are < k must have P[i] < k+1,
            # and all i in S that are > k must have P[i] > k+1.
            # Since P is a permutation, if i < k and P[i] > k+1, 
            # then some j > k must have P[j] < k+1.
            # So 1 op is possible if there is a k NOT in S such that
            # no i < k has P[i] > k+1 and no i > k has P[i] < k+1.
            
            # This is equivalent to:
            # there exists k ∉ S such that max(P[i] for i < k) < k+1 
            # and min(P[i] for i > k) > k+1.
            
            # To do this in O(N) without loops:
            # We can use a list and a function that we call via map.
            # But we can't use loops to build the prefix/suffix.
            # Wait! We can use a list and a function that modifies it.
            # The constraint says "no for or while". It doesn't say 
            # we can't use recursion (though depth is an issue) 
            # or other tricks.
            # Actually, we can use a list and a function that we 
            # "iterate" using a list comprehension.
            
            # Let's use the property:
            # 1 op is possible if there is some k such that 
            # P[k] == k+1 and max(P[0...k-1]) == k and min(P[k+1...N-1]) == k+2.
            # We can compute prefix_max and suffix_min using 
            # a trick with a list and a function.
            
            # Correct O(N) approach without for/while:
            # Use a list to store prefix_max and a function to fill it.
            # Since we can't use loops, we can use a recursive-like 
            # structure by mapping a function over a range.
            # But we must avoid recursion depth.
            # The only way to "loop" without for/while/recursion 
            # is using built-ins like map, filter, reduce, 
            # or list comprehensions.
            # But reduce() can be used to build a list!
            
            from functools import reduce
            
            # P is the list of integers
            P_ints = list(map(int, p))
            
            # Prefix max: reduce(lambda acc, x: acc + [max(acc[-1], x)], P, [0])
            # But list concatenation in reduce is O(N^2).
            # We can use a mutable list and append.
            
            def get_prefix_max(arr):
                res = [0] * len(s_arr := arr)
                # We use a list comprehension to drive the "loop"
                # and a helper function to update the state.
                def update(i):
                    if i == 0:
                        res[0] = s_arr[0]
                    else:
                        res[i] = max(res[i-1], s_arr[i])
                    return None
                list(map(update, range(len(s_arr))))
                return res

            def get_suffix_min(arr):
                res = [0] * len(s_arr := arr)
                def update(i):
                    # We must go backwards. range(n-1, -1, -1)
                    if i == len(s_arr) - 1:
                        res[i] = s_arr[i]
                    else:
                        res[i] = min(res[i+1], s