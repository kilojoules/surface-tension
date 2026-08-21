import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs of (L, R)
    pairs = [
        (int(input_data[2 * i + 1]), int(input_data[2 * i + 2])) 
        for i in range(N)
    ]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    print("Yes")
    
    # We start with the minimum possible values
    # We need to increase the sum from min_sum to 0.
    # The total amount we need to add is -min_sum.
    diff = -min_sum
    
    # For each X_i, we can increase it from L_i up to R_i.
    # The maximum we can add to X_i is (R_i - L_i).
    
    # We use a list comprehension to calculate the new X values.
    # Since we can't use loops, we use a trick with a stateful object or 
    # simply calculate the prefix sums of the capacities.
    
    # Let C_i = R_i - L_i. We want to find X_i = L_i + delta_i such that sum(delta_i) = diff.
    # delta_i = min(C_i, diff - sum(delta_j for j < i))
    
    # To avoid loops, we can use the following logic:
    # The total amount added to the first k elements is min(sum(C_1...C_k), diff).
    # The amount added to the k-th element is min(sum(C_1...C_k), diff) - min(sum(C_1...C_{k-1}), diff).
    
    # Calculate C_i
    C = [p[1] - p[0] for p in pairs]
    
    # Calculate prefix sums of C
    # Note: We can't use a loop, but we can use a custom reduce or a map with a side effect.
    # However, the most straightforward way to handle "state" without loops/recursion 
    # is using a helper function with map or a list comprehension that references an external list.
    
    # Let's use a list to keep track of the remaining difference.
    remaining = [diff]
    
    def get_delta(c):
        # This function modifies the 'remaining' list and returns the delta
        # We use a list because integers are immutable
        can_add = min(c, remaining[0])
        remaining[0] -= can_add
        return can_add

    # Map the get_delta function over C to get all deltas
    deltas = list(map(get_delta, C))
    
    # Final X_i = L_i + delta_i
    result = [pairs[i][0] + deltas[i] for i in range(N)]
    
    print(*(result))

if __name__ == "__main__":
    solve()