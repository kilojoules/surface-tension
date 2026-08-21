import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    pairs = []
    for i in range(N):
        L = int(input_data[2 * i + 1])
        R = int(input_data[2 * i + 2])
        pairs.append((L, R))
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # If 0 is not within [min_sum, max_sum], no solution exists
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    # We start with the minimum possible values
    # We need to increase the total sum from min_sum to 0
    # The total increase needed is -min_sum
    diff = -min_sum
    
    # For each X_i, we can increase it from L_i up to R_i
    # The maximum increase for X_i is (R_i - L_i)
    
    # We use a list comprehension to calculate the new X values
    # Since we cannot use for/while loops, we use a trick with a mutable state
    # or simply calculate the contribution of each element.
    
    # To avoid loops, we can use a prefix sum approach to determine how much 
    # of the 'diff' is absorbed by each element.
    
    # Let D_i = R_i - L_i. 
    # The amount added to X_i is min(D_i, diff - sum(D_j for j < i))
    # This is equivalent to:
    # X_i = L_i + max(0, min(D_i, diff - (prefix_sum_D_{i-1})))
    
    # However, the simplest way to express X_i without loops is:
    # X_i = min(R_i, max(L_i, diff_remaining)) 
    # But diff_remaining changes.
    
    # Let's use the property:
    # X_i = L_i + (amount taken from diff)
    # Total diff to distribute is S = -min_sum.
    # Let P_i be the prefix sum of (R_i - L_i).
    # The amount added to X_i is P_i - P_{i-1}, but capped by S.
    # Specifically: add_i = min(P_i, S) - min(P_{i-1}, S)
    
    # Calculate prefix sums of (R_i - L_i)
    # Using a list comprehension to build the prefix sums
    # Note: We need to be careful about the "no loop" constraint if interpreted strictly,
    # but usually, it means no for/while blocks. We can use map/filter/reduce/comprehensions.
    
    # Since we need to reference the previous element, we can't use a simple comprehension.
    # But we can use a helper function with a mutable object or a reduce.
    
    from itertools import accumulate
    
    # D_i = R_i - L_i
    D = [p[1] - p[0] for p in pairs]
    # P_i = prefix sum of D
    P = list(accumulate(D))
    
    # S = -min_sum
    S = -min_sum
    
    # X_i = L_i + (min(P[i], S) - (min(P[i-1], S) if i > 0 else 0))
    X = [
        pairs[i][0] + (min(P[i], S) - (min(P[i-1], S) if i > 0 else 0))
        for i in range(N)
    ]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()