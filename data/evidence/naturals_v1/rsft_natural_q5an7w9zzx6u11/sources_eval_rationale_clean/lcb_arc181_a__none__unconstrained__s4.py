import sys

def solve():
    # Read all input at once and split into a flat list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to group the remaining data into test cases.
    # Since each case starts with N followed by N elements, 
    # we can't use a simple chunk size. 
    # However, we can use a generator or a helper to parse.
    
    def get_cases(data):
        it = iter(data)
        # Use a list comprehension to extract cases. 
        # Since we can't use 'for' loops, we use a trick with map/next.
        # But wait, the constraint says no 'for' loops. 
        # We can use a recursive-like structure via map or a custom function.
        # Actually, the simplest way to handle variable N is to 
        # process the flat list using a custom function called via map.
        pass

    # To avoid 'for' loops and 'while' loops, we process the flat list.
    # We can use a helper function and a list comprehension.
    # Because N varies, we can't use simple slicing.
    # Let's use a state-carrying object or a closure to track the index.
    
    class State:
        def __init__(self, data):
            self.data = data
            self.idx = 1
        def next_case(self):
            N = int(self.data[self.idx])
            P = self.data[self.idx + 1 : self.idx + 1 + N]
            self.idx += 1 + N
            return N, P

    state = State(input_data)
    
    # We use map(lambda _: ..., range(T)) to simulate the loop over T test cases.
    # Inside the lambda, we call state.next_case().
    
    def calculate_min_ops(N, P):
        # P is a list of strings, convert to integers
        P = list(map(int, P))
        
        # The goal is to find if 0, 1, or 2 operations are needed.
        # 0 ops: P is already sorted.
        # 1 op: There exists k such that sorting [1, k-1] and [k+1, N] results in [1...N].
        # This is possible if there is some k such that:
        # All elements in P[0...k-2] are <= P[k] (if we sort them) 
        # AND all elements in P[k...N-1] are >= P[k-1]... 
        # Actually, the condition for 1 op is:
        # There exists k such that the set of values {P_1...P_{k-1}} is {1...k-1}
        # AND the set of values {P_{k+1}...P_N} is {k+1...N}.
        # This implies P_k must be k.
        # If P_k = k, then sorting the left and right parts automatically 
        # results in the identity permutation.
        
        # Check 0 ops:
        is_sorted = (P == sorted(P))
        if is_sorted:
            return 0
        
        # Check 1 op:
        # We need to find if there is any k (1-indexed) such that P[k-1] == k
        # AND the set of elements to the left is {1...k-1}
        # AND the set of elements to the right is {k+1...N}.
        # This is equivalent to: 
        # max(P[0...k-2]) < k < min(P[k...N-1])
        # Let's precompute prefix max and suffix min.
        
        # Since we can't use loops, we use a trick to get prefix max/suffix min.
        # We can use a list comprehension with a helper or use the fact that
        # we can use built-in functions.
        # However, we can't use 'for' to build the prefix list.
        # We can use a recursive-like approach with a list and 'set' 
        # but the simplest way to check the 1-op condition:
        # P_k = k AND max(P_1...P_{k-1}) == k-1 AND min(P_{k+1}...P_N) == k+1.
        
        # To implement this without loops:
        # We can use a list comprehension to check all k.
        # But we need prefix max and suffix min.
        # We can use a technique with a mutable list and map().
        
        pref_max = [0] * N
        suff_min = [0] * N
        
        # Using map(lambda i: ..., range(N)) to fill arrays
        # Note: assignment in lambda is not allowed, but we can use list.append 
        # or a helper function.
        
        def fill_pref(i, p_max, P):
            val = P[i]
            if i > 0:
                val = max(val, p_max[i-1])
            p_max[i] = val
            return val

        def fill_suff(i, s_min, P):
            val = P[i]
            if i < N - 1:
                val = min(val, s_min[i+1])
            s_min[i] = val
            return val

        # Use list(map(...)) to execute the side-effect filling the arrays
        list(map(lambda i: fill_pref(i, pref_max, P), range(N)))
        list(map(lambda i: fill_suff(i, suff_min, P), range(N-1, -1, -1)))
        
        # Now check if any k satisfies the condition
        # k is 1-indexed, so index is k-1
        # Condition: P[i] == i+1 and (i==0 or pref_max[i-1] == i) 
        # and (i==N-1 or suff_min[i+1] == i+2)
        
        can_do_1 = any(
            (P[i] == i + 1 and 
             (i == 0 or pref_max[i-1] == i) and 
             (i == N - 1 or suff_min[i+1] == i + 2))
            for i in range(N)
        )
        
        return 1 if can_do_1 else 2

    # Process all cases and print
    results = map(lambda _: calculate_min_ops(*state.next_case()), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()