import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data
    # Since we cannot use loops, we process the flat list using a 
    # combination of slicing and map/comprehensions.
    
    # We need to group the input into test cases.
    # Each case starts with N, followed by N elements.
    # To avoid loops, we can pre-calculate the starting indices of each case.
    
    # However, a simpler way to avoid loops while processing T cases 
    # is to use a recursive-like structure via map or a comprehension
    # if we can determine the boundaries.
    
    # Let's build a list of (N, P) pairs.
    # Since we can't use while loops, we use a trick with a generator 
    # or just process the list based on the known structure.
    
    def get_cases(data):
        # This is a tricky part without loops. 
        # We can use a list comprehension if we know the indices.
        # But N varies. We can use a custom function with map.
        pass

    # Actually, the most reliable way to avoid loops and recursion 
    # for variable-length chunks is to use an iterator.
    it = iter(input_data)
    
    def process_case():
        try:
            N = int(next(it))
            P = [int(next(it)) for _ in range(N)]
            
            # Check if already sorted
            # We can't use loops, so we use all() and list comprehensions.
            is_sorted = all(P[i] == i + 1 for i in range(N))
            if is_sorted:
                return 0
            
            # Check if 1 operation is enough.
            # 1 operation with index k works if:
            # The elements in P[0...k-2] are a permutation of 1...k-1
            # AND the elements in P[k...N-1] are a permutation of k+1...N.
            # This is true if max(P[0...k-2]) == k-1 and P[k-1] == k.
            
            # Precompute prefix maximums
            # Since we can't use loops, we use a trick to get prefix maxes.
            # But wait, the constraint says "no for/while loops". 
            # List comprehensions are allowed. 
            # But prefix max requires the previous value.
            # We can use itertools.accumulate.
            from itertools import accumulate
            prefix_max = list(accumulate(P, max))
            
            # Condition for k (1-indexed):
            # For k=1: P[0] is ignored, P[1...N-1] sorted. 
            # This is possible if P[0] == 1 is NOT required, 
            # but the operation sorts P[1...N-1].
            # Actually, the operation is: sort 1 to k-1, sort k+1 to N.
            # If k=1, we sort P[1...N-1]. This works if P[0] is already 1.
            # Wait, the sample says k=3 for (2,1,3,5,4) works.
            # P[0]=2, P[1]=1, P[2]=3, P[3]=5, P[4]=4.
            # k=3: sort P[0...1] -> (1,2), sort P[3...4] -> (4,5).
            # Result: (1,2,3,4,5).
            # So the condition is: P[k-1] == k and 
            # max(P[0...k-2]) == k-1 (if k > 1) and
            # min(P[k...N-1]) == k+1 (if k < N).
            
            # Let's use prefix_max and suffix_min.
            from itertools import accumulate
            p_max = list(accumulate(P, max))
            s_min = list(accumulate(P[::-1], min))[::-1]
            
            # Check if any k in 1...N satisfies the condition.
            # k is 1-indexed.
            # For k:
            # 1. If k > 1, p_max[k-2] must be k-1.
            # 2. P[k-1] must be k.
            # 3. If k < N, s_min[k] must be k+1.
            
            # We use a list comprehension to check all k and then max() or any().
            can_do_1 = any(
                (k == 1 or p_max[k-2] == k-1) and 
                (P[k-1] == k) and 
                (k == N or s_min[k] == k+1)
                for k in range(1, N + 1)
            )
            
            return 1 if can_do_1 else 2
        except StopIteration:
            return None

    # To avoid the loop for T cases, we use map and a list.
    # Since process_case uses next(it), we can just call it T times.
    # But we can't use a loop. We can use map(lambda _, __: process_case(), range(T)).
    results = map(lambda _: process_case(), range(T))
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    solve()