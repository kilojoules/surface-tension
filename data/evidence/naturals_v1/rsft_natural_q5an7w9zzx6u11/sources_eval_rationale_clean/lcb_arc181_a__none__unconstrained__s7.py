import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to process the remaining data in chunks of N + 1
    # Since we cannot use loops, we will use a generator or map 
    # to handle the test cases.
    
    # To avoid loops, we first determine the boundaries of each test case.
    # We can use a recursive-like structure via map or a comprehension.
    # However, since N varies, we need a way to slice the input_data.
    
    # Let's pre-calculate the starting indices of each test case.
    # We can't use a loop to find indices, so we use a trick with 
    # a list and a helper function or a comprehension.
    # Actually, the simplest way to avoid 'for' and 'while' is to 
    # use a recursive-style processing via a list comprehension 
    # if we can index it, but we can't easily.
    
    # Wait, the constraint says "no for or while loops". 
    # We can use map() and list comprehensions.
    # To handle variable N, we can use a custom function and 
    # pass it to map, but we need to track the current index.
    # Since we can't use a loop to iterate through T, 
    # we can use a recursive function (though Python has recursion limits)
    # or we can process the flat list using a clever indexing trick.
    
    # Let's use a helper function that processes the list and 
    # returns the results, called via a method that doesn't use loops.
    
    def process_cases(data, t_remaining, current_idx):
        if t_remaining == 0:
            return []
        
        N = int(data[current_idx])
        P = data[current_idx + 1 : current_idx + 1 + N]
        
        # Logic to find min operations:
        # 0 ops: already sorted
        # 1 op: there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        # This happens if there is some k where P[k] is the only element 
        # that could be "out of place" and the others can be partitioned.
        # Actually, 1 op is possible if there exists k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} and {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # This is equivalent to saying P_k = k and the prefix/suffix are permutations 
        # of their respective ranges.
        # Wait, the operation is: sort 1 to k-1 AND sort k+1 to N.
        # After one operation, the array becomes sorted if and only if 
        # the set of elements {P_1, ..., P_{k-1}} is {1, ..., k-1} 
        # AND the set of elements {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # This implies P_k must be k.
        
        # Let's refine: 1 op is possible if there exists k such that:
        # max(P_1 ... P_{k-1}) < k and min(P_{k+1} ... P_N) > k.
        # This is true if P_k = k and max(P_1 ... P_{k-1}) = k-1.
        
        # To check this without loops:
        # Let pref_max[i] be max of first i elements.
        # Let suff_min[i] be min of elements from i to N.
        # 1 op is possible if there's an i where pref_max[i-1] == i-1 and suff_min[i+1] == i+1.
        
        # But we can't use loops to build pref_max. 
        # We can use itertools.accumulate.
        
        # However, the constraint to avoid loops is strict. 
        # Let's use a different approach for the T cases.
        # We can use a recursive function and sys.setrecursionlimit.
        pass

# Since I cannot use loops, I will use map, list comprehensions, 
# and itertools.accumulate to handle the logic.
# To handle the T test cases without a loop, I will use a 
# recursive function to consume the input list.

import itertools

def solve_all():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a recursive function to process cases
    def run(idx, t_left):
        if t_left == 0:
            return []
        
        N = int(input_data[idx])
        P = list(map(int, input_data[idx + 1 : idx + 1 + N]))
        
        # Check if 0 operations needed
        is_sorted = (P == sorted(P))
        
        # Check if 1 operation needed
        # pref_max[i] = max(P[0...i])
        # suff_min[i] = min(P[i...N-1])
        pref_max = list(itertools.accumulate(P, max))
        suff_min = list(itertools.accumulate(P[::-1], min))[::-1]
        
        # k is 1-indexed in problem, 0-indexed in P is k-1.
        # For a given k (0-indexed), we need:
        # (k == 0 or pref_max[k-1] == k) AND (k == N-1 or suff_min[k+1] == k+2)
        # AND P[k] == k+1
        
        # We use a list comprehension to check all k and 'any()' to see if any work.
        can_do_1 = any(
            (k == 0 or pref_max[k-1] == k) and 
            (k == N-1 or suff_min[k+1] == k+2) and 
            (P[k] == k+1)
            for k in range(N)
        )
        
        # The problem says it's always possible. 
        # If not 0 and not 1, the answer must be 2.
        # (Because we can pick k=1 to sort [2, N], then k=N to sort [1, N-1])
        # Actually, 2 operations are always sufficient for N >= 3.
        
        res = 0 if is_sorted else (1 if can_do_1 else 2)
        
        return [res] + run(idx + 1 + N, t_left - 1)

    # To avoid recursion depth issues and loops, 
    # we can't use 'for' to iterate T. 
    # But we can use a list comprehension to drive the recursion 
    # if we wrap the state in a list.
    # Actually, the most reliable way to avoid 'for' is to use 
    # a generator and map.
    
    # Let's redefine the logic to be used inside a map.
    def get_result(case_tuple):
        N, P = case_tuple
        if P == sorted(P):
            return 0
        pref_max = list(itertools.accumulate(P, max))
        suff_min = list(itertools.accumulate(P[::-1], min))[::-1]
        if any((k == 0 or pref_max[k-1] == k) and 
               (k == N-1 or suff_min[k+1] == k+2) and 
               (P[k] == k+1) for k in range(N)):
            return 1
        return 2

    # To split the flat input_data into (N, P) pairs without loops:
    # We can use a recursive function to build the list of cases.
    def build_cases(idx, t_left):
        if t_left == 0:
            return []
        N = int(input_data[idx])
        P = list(map(int, input_data[idx + 1 : idx + 1 + N]))
        return [(N, P)] + build_cases(idx + 1 + N, t_left - 1)

    # Since T can be 10^5, recursion will hit limit.
    # But wait, I can use a list comprehension with a 
    # custom iterator or a trick.
    # The only way to process a variable-length stream without 
    # for/while is using map/filter/reduce or recursion.
    # For T=10^5, I must use sys.setrecursionlimit.
    
    sys.setrecursionlimit(200000)
    cases = build_cases(1, T)
    results = map(get_result, cases)
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve_all()