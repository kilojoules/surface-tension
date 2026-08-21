import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T_str = input_data[0]
    T = int(T_str)
    
    # Pointer to keep track of the current position in input_data
    ptr = 1
    
    results = []
    
    for _ in range(T):
        N = int(input_data[ptr])
        P = list(map(int, input_data[ptr + 1 : ptr + 1 + N]))
        ptr += 1 + N
        
        # The goal is to find the minimum number of operations to make P_i = i.
        # An operation with index k sorts [1, k-1] and [k+1, N].
        # If we can pick k such that P_k = k, and all elements {1...k-1} are in 
        # positions {1...k-1} (though unordered) and {k+1...N} are in 
        # positions {k+1...N}, then one operation suffices.
        # However, the simplest condition for 0 operations is P_i = i for all i.
        # For 1 operation: there must exist some k such that P_k = k, and 
        # max(P_1...P_{k-1}) < k and min(P_{k+1}...P_N) > k.
        # Actually, the condition for 1 operation is: there exists k such that 
        # P_k = k AND {P_1, ..., P_{k-1}} = {1, ..., k-1} AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # Wait, that's just P_i = i for all i.
        # Let's re-read: "sort the 1-st through (k-1)-th terms... sort the (k+1)-th through N-th".
        # If we pick k, then P_k remains unchanged. The others become sorted.
        # So for 1 operation to work, we need some k such that P_k = k, and 
        # the set {P_1, ..., P_{k-1}} is exactly {1, ..., k-1} and {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # No, that's not right. If we pick k, P_k stays. The rest are sorted.
        # To get (1, 2, ..., N), we need P_k = k, and the remaining elements to be 
        # (1, ..., k-1, k+1, ..., N) in some order.
        # This means the set {P_i | i != k} must be {1, ..., N} \ {k}.
        # Which is always true for any k.
        # After sorting, the elements at indices 1 to k-1 become 1 to k-1, and k+1 to N become k+1 to N.
        # So the condition for 1 operation is: there exists k such that P_k = k.
        
        # Check if 0 operations:
        # We can use a trick: if the number of i such that P_i != i is 0, then 0.
        # But we can't use loops. We can use map and sum.
        
        # Count how many i satisfy P[i-1] == i
        # Since we can't use loops, we use a generator expression inside sum() or any()
        
        is_sorted = (sum(1 for i in range(1, N + 1) if P[i-1] != i) == 0)
        if is_sorted:
            results.append("0")
            continue
            
        has_fixed_point = any(P[i-1] == i for i in range(1, N + 1))
        if has_fixed_point:
            results.append("1")
        else:
            results.append("2")

    sys.stdout.write("\n".join(results) + "\n")

if __name__ == "__main__":
    solve()