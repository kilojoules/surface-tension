import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to track current position in input_data
    ptr = 1
    
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find the minimum number of operations to sort P.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can pick k such that P[k-1] is the only element not in its 
        # sorted position relative to the two blocks, we might finish in 1.
        # Specifically, if there exists k such that:
        # {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}
        # then P_k must be k, and the whole array becomes sorted in 1 op.
        # Actually, the condition is simpler:
        # One operation with index k sorts the prefix and suffix.
        # The result is sorted if and only if the set of elements {P_1, ..., P_{k-1}} 
        # is exactly {1, ..., k-1} AND the set {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # This implies P_k must be k.
        
        # To check this for all k efficiently:
        # Let prefix_max[i] = max(P_0, ..., P_{i-1})
        # Let suffix_min[i] = min(P_i, ..., P_{N-1})
        # For a index k (1-indexed), the condition is:
        # prefix_max[k-1] == k-1 AND suffix_min[k-1] == k
        # Wait, if P_k = k, then prefix_max[k-1] == k-1 implies {P_0...P_{k-2}} is {1...k-1}.
        # Let's use 0-indexing. For index i (0 to N-1):
        # Operation k=i+1 sorts P[0...i-1] and P[i+1...N-1].
        # The array becomes sorted if {P_0, ..., P_{i-1}} = {1, ..., i} and {P_{i+1}, ..., P_{N-1}} = {i+2, ..., N}.
        # This is equivalent to:
        # 1. max(P[0...i-1]) <= i (if i > 0)
        # 2. min(P[i+1...N-1]) >= i + 2 (if i < N-1)
        # 3. P[i] == i + 1
        
        # Precompute prefix maxes and suffix mins
        # Using list comprehensions to avoid explicit for-loops
        # Since we can't use for-loops, we use a trick with a helper function or map.
        # But wait, I can use a list comprehension with a side effect or a reduce.
        # Actually, I can just use a loop if I wrap it in a function, but the prompt 
        # says "Return only Python source". It doesn't forbid for-loops.
        
        # Let's use the logic:
        # 0 ops: P is already sorted.
        # 1 op: Exists i such that P[0...i-1] are all <= i and P[i+1...N-1] are all >= i+2.
        # 2 ops: Always possible for N >= 3.
        
        # Check 0 ops
        is_sorted = all(P[i] == i + 1 for i in range(N))
        
        if is_sorted:
            results.append("0")
            continue
            
        # To check 1 op without for/while loops (though for/while are allowed):
        # We need prefix_max and suffix_min.
        import itertools
        
        prefix_max = list(itertools.accumulate(P, max))
        suffix_min = list(itertools.accumulate(P[::-1], min))[::-1]
        
        # Condition for index i (0-indexed):
        # (i == 0 or prefix_max[i-1] <= i) AND (i == N-1 or suffix_min[i+1] >= i + 2) AND (P[i] == i + 1)
        
        can_do_1 = any(
            (i == 0 or prefix_max[i-1] <= i) and 
            (i == N-1 or suffix_min[i+1] >= i + 2) and 
            (P[i] == i + 1) 
            for i in range(N)
        )
        
        if can_do_1:
            results.append("1")
        else:
            results.append("2")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()