import sys
from functools import reduce

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to process test cases
    # Each case starts at index 1 + sum of (1 + N_i) for previous cases
    # To avoid loops, we can use a recursive-like structure or map
    # But since we need to track the current index in the flat input_data list,
    # we will group the data first.
    
    def get_cases(data):
        # This helper groups the flat list into (N, P) pairs
        # Using a custom reduce to partition the list
        def accumulate(acc, current_idx):
            N = int(data[current_idx])
            P = data[current_idx + 1 : current_idx + 1 + N]
            return acc + [(N, P)], current_idx + 1 + N
        
        # We can't use a loop to call accumulate, so we use a trick with range and reduce
        # However, the constraint is to avoid for/while. 
        # We can use a recursive-style approach via a list comprehension if we know the number of cases.
        # But the most reliable way to handle variable N is to process the list.
        pass

    # Since we cannot use loops, we use a recursive-like approach via a 
    # helper function and map, but we must be careful with recursion limits.
    # Instead, we can use a technique to slice the list based on the N values.
    
    def process_all(data):
        # We use a list to store the results and a helper to find the boundaries
        # Because we can't use loops, we use a trick: 
        # 1. Identify the start index of each test case.
        # 2. Use those indices to slice the data.
        
        # To find indices without a loop, we can use a custom reduce.
        # The state is (list_of_indices, current_index)
        def find_indices(state, _):
            indices, curr = state
            if curr >= len(data):
                return (indices, curr)
            N = int(data[curr])
            return (indices + [curr], curr + 1 + N)
        
        # We need to run this T times.
        # Since we can't use for-loops, we use map(None, range(T)) and reduce.
        # But wait, the constraint says "no for or while loops". 
        # We can use map, filter, reduce, and comprehensions.
        
        # Let's refine the logic:
        # The answer is:
        # 0 if P is already sorted.
        # 1 if there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        #   This happens if there is some k such that:
        #   {P_1, ..., P_{k-1}, P_{k+1}, ..., P_N} = {1, ..., N} \ {k}
        #   AND the elements that end up in positions 1...k-1 are indeed 1...k-1
        #   AND the elements that end up in positions k+1...N are k+1...N.
        #   Actually, the operation sorts the two ranges. 
        #   The only element that doesn't move is P_k.
        #   For the result to be sorted, we must have P_k = k, and the set of 
        #   remaining elements must be {1, ..., k-1, k+1, ..., N}.
        #   Since P is a permutation, P_k = k is sufficient to make the 
        #   remaining elements the correct set.
        #   After sorting, the array becomes sorted if and only if 
        #   all elements in positions 1...k-1 are < k and all in k+1...N are > k.
        #   This is true if and only if P_k = k.
        # 2 otherwise (it is proven that 2 is always enough).
        
        # Wait, the logic "P_k = k" is for 1 operation. 
        # If P_k = k, then sorting [1, k-1] and [k+1, N] will result in 
        # [1, 2, ..., k-1, k, k+1, ..., N].
        # So the answer is 0 if sorted, 1 if there exists k such that P_k = k, 2 otherwise.
        # BUT, there is a catch: the operation sorts 1 to k-1 and k+1 to N.
        # If k=1, it sorts 2 to N. If k=N, it sorts 1 to N-1.
        # If P = [2, 1, 3], k=3 sorts [2, 1] -> [1, 2, 3]. Correct.
        # If P = [3, 2, 1], k=2 sorts [3] and [1] -> [3, 2, 1]. Incorrect.
        # So the condition for 1 operation is: there exists k such that P_k = k.
        # Let's check Sample 3: [3, 2, 1, 7, 5, 6, 4]. 
        # P_1=3, P_2=2, P_3=1, P_4=7, P_5=5, P_6=6, P_7=4.
        # P_2=2, P_5=5, P_6=6. All these k satisfy P_k=k.
        # But the sample says the answer is 2. Why?
        # Re-read: "sort the 1-st through (k-1)-th terms... sort the (k+1)-th through N-th terms".
        # In Sample 3: P = [3, 2, 1, 7, 5, 6, 4].
        # If k=2: sort P[1:1] and P[3:7]. P becomes [3, 2, 1, 4, 5, 6, 7]. Not sorted.
        # If k=5: sort P[1:4] and P[6:7]. P becomes [1, 2, 3, 7, 5, 4, 6]. Not sorted.
        # The condition for 1 operation is: there exists k such that 
        # sorting P[1...k-1] and P[k+1...N] results in [1...N].
        # This happens if and only if:
        # 1. P_k = k
        # 2. {P_1, ..., P_{k-1}} = {1, ..., k-1}
        # 3. {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # Condition 2 is equivalent to: max(P_1, ..., P_{k-1}) = k-1.
        # Condition 3 is equivalent to: min(P_{k+1}, ..., P_N) = k+1.
        
        # Let's re-evaluate:
        # For a fixed k, the operation succeeds if:
        # (k == 1 or max(P[0:k-1]) == k-1) AND 
        # (k == N or min(P[k:N]) == k+1) AND 
        # (P[k-1] == k)
        
        # To implement this without loops:
        # 1. Compute prefix maximums.
        # 2. Compute suffix minimums.
        # 3. Check the condition for each k using a list comprehension.
        
        def solve_case(N, P):
            # P is 0-indexed, so P_i is P[i-1]
            # Prefix max
            # Since we can't use loops, we use a trick to get prefix maxes.
            # In Python, we can't use itertools.accumulate? No, the prompt says 
            # "no for or while loops", but doesn't forbid imports.
            # However, I'll use a list comprehension with a helper if needed.
            # Actually, itertools.accumulate is perfect.
            from itertools import accumulate
            
            pref_max = list(accumulate(P, max))
            suff_min = list(accumulate(P[::-1], min))[::-1]
            
            # Condition for k (1-indexed):
            # k=1: P[0]==1 and suff_min[1]==2
            # k=N: P[N-1]==N and pref_max[N-2]==N-1
            # 1 < k < N: P[k-1]==k and pref_max[k-2]==k-1 and suff_min[k]==k+1
            
            # We can generalize:
            # For k in 1...N:
            # left_ok = (k == 1 or pref_max[k-2] == k-1)
            # right_ok = (k == N or suff_min[k] == k+1)
            # mid_ok = (P[k-1] == k)
            
            # Check if already sorted
            if P == sorted(P):
                return 0
            
            # Check if 1 operation suffices
            # Use a list comprehension to check all k
            can_do_1 = any([
                ( (k == 1 or pref_max[k-2] == k-1) and 
                  (k == N or suff_min[k] == k+1) and 
                  (P[k-1] == k) )
                for k in range(1, N + 1)
            ])
            
            return 1 if can_do_1 else 2

        # To handle the T test cases without loops:
        # 1. Parse the flat list into a list of (N, P)
        # 2. Map solve_case over them.
        
        # Since we can't use loops to parse, we use a recursive-like 
        # approach or a clever reduce to group the input.
        def group_input(data):
            # data[0] is T
            T = int(data[0])
            # We use a helper to get the start indices
            def get_indices(acc, curr):
                if curr >= len(data): return acc
                N = int(data[curr])
                return get_indices(acc + [curr], curr + 1 + N)
            
            # But recursion limit is an issue. Let's use a different way.
            # We can use a generator and next() inside a list comprehension.
            # But we can't use a loop to call next().
            # Wait, we can use a generator and map!
            
            def gen_cases(it):
                # This is a generator, but we can't use 'for' inside it.
                # We can use recursion, but that's risky.
                # Actually, we can use a while-loop inside a generator? 
                # No, "no for or while loops".
                pass
        
        # Let's use a different approach to parse:
        # Since sum of N is 2e5, we can't use deep recursion.
        # We can use a list comprehension with a mutable state (a list) 
        # and a helper function.
        
        state = {'curr': 1}
        def extract_case(data):
            N = int(data[state['curr']])
            P = list(map(int, data[state['curr']+1 : state['curr']+1+N]))
            state['curr'] += 1 + N
            return (N, P)
        
        # Use map to call extract_case T times
        results = map(lambda _: solve_case(*extract_case(data)), range(T))
        return "\n".join(map(str, results))

    # To avoid the 'def' inside 'def' and the state dictionary 
    # which might be frowned upon, I'll flatten it.
    # But the logic remains the same.
    print(process_all(input_data))

# Since I need to provide a single block, I will integrate everything.