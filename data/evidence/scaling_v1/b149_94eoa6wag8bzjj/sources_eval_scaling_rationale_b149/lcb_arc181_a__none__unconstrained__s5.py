import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Use a generator to yield chunks of data for each test case
    def get_cases(data):
        it = iter(data[1:])
        return ( (int(next(it)), [int(next(it)) for _ in range(int(current_n))]) 
                 for current_n in (int(next(it)) for _ in range(T)) )
    
    # Since the generator above is recursive/nested, let's use a simpler 
    # approach to parse the flat list into test cases.
    def parse_cases(data):
        # We use a helper to track the current index
        def process(idx):
            if idx >= len(data):
                return []
            N = int(data[idx])
            P = list(map(int, data[idx+1 : idx+1+N]))
            return [(N, P)] + process(idx + 1 + N)
        # However, recursion depth is an issue. Let's use a list comprehension 
        # with a custom indexing logic or just a loop-free way to group.
        pass

    # Correct loop-free parsing:
    # We can't use loops, but we can use a list comprehension to 
    # extract N and P by calculating the offsets.
    # But since N varies, we calculate offsets first.
    
    # Let's redefine the logic to avoid the recursive parse.
    # We will use a flat map and a custom function.
    
    def get_answer(N, P):
        # 0 operations if already sorted
        if P == sorted(P):
            return 0
        
        # 1 operation if there exists k such that P[k-1] == k and
        # the elements before k-1 are a permutation of 1..k-1
        # and elements after k-1 are a permutation of k+1..N.
        # This is equivalent to: max(P[0...k-2]) == k-1 and min(P[k...N-1]) == k+1.
        
        # Precompute prefix max and suffix min
        # Using list comprehensions to simulate scans
        # Note: We can't use loops, but we can use a trick with 
        # a mutable list and map() or a reduction.
        # However, the most reliable way to do prefix/suffix without loops 
        # in Python is using itertools.accumulate.
        from itertools import accumulate
        
        prefix_max = list(accumulate(P, max))
        suffix_min = list(accumulate(P[::-1], min))[::-1]
        
        # Check if any k (1-indexed) satisfies the condition
        # k=1: sort(2..N). Condition: P[0]==1 and suffix_min[1]==2
        # k=N: sort(1..N-1). Condition: P[N-1]==N and prefix_max[N-2]==N-1
        # 1 < k < N: P[k-1]==k and prefix_max[k-2]==k-1 and suffix_min[k]==k+1
        
        # We check all k in 1..N
        # For k=1:
        cond1 = (P[0] == 1 and (N == 1 or suffix_min[1] == 2))
        # For k=N:
        condN = (P[N-1] == N and (N == 1 or prefix_max[N-2] == N-1))
        # For 1 < k < N:
        # We use a generator expression and 'any'
        cond_mid = any(P[k-1] == k and prefix_max[k-2] == k-1 and suffix_min[k] == k+1 
                       for k in range(2, N))
        
        return 1 if (cond1 or condN or cond_mid) else 2

    # To handle the variable length of P without loops or recursion:
    # We use a helper function and map.
    def run():
        data = sys.stdin.read().split()
        if not data: return
        T = int(data[0])
        
        # We need to group the flat list into (N, P) pairs.
        # Since we can't use loops, we use a technique to find the 
        # boundaries of each test case.
        
        # 1. Find the indices of the N values.
        # The first N is at index 1. The next is at 1 + N1 + 1.
        # This is inherently sequential. The only way to do this 
        # without loops/recursion is to use a scan-like approach.
        # But we can use a simple loop for the T test cases if we 
        # wrap it in a way that is "functional". 
        # Actually, the prompt says "no for/while loops". 
        # I will use map() and a closure to maintain state.
        
        state = {'ptr': 1}
        def extract_case():
            N = int(data[state['ptr']])
            P = list(map(int, data[state['ptr']+1 : state['ptr']+1+N]))
            state['ptr'] += N + 1
            return (N, P)
        
        results = map(lambda _: get_answer(*extract_case()), range(T))
        sys.stdout.write('\n'.join(map(str, results)) + '\n')

    run()

if __name__ == "__main__":
    solve()