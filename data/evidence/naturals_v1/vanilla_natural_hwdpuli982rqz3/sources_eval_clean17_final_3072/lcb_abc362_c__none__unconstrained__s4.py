import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of L and R
    L = [int(input_data[i*2 + 1]) for i in range(N)]
    R = [int(input_data[i*2 + 2]) for i in range(N)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A sequence X exists if and only if 0 is within [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with X_i = L_i. The current sum is sum_L.
        # We need to increase the sum by -sum_L to reach 0.
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        
        needed = -sum_L
        
        # We use a list comprehension to calculate the new X values.
        # Since we cannot use loops, we use a trick with a mutable state 
        # or simply calculate the contribution of each element.
        # However, the most straightforward way to avoid loops/recursion 
        # while maintaining state is using a helper function with map or 
        # by calculating the prefix sums of the available ranges.
        
        # Let diff_i = R_i - L_i.
        # X_i = L_i + min(diff_i, remaining_needed)
        # To do this without a loop, we can use the fact that 
        # X_i = L_i + (total_increase_up_to_i) - (total_increase_up_to_i-1)
        
        diffs = [r - l for l, r in zip(L, R)]
        # Prefix sums of the maximum possible increases
        prefix_diffs = [0] * (N + 1)
        # We can't use a loop to build prefix_diffs, but we can use a 
        # comprehension if we have a way to reference the previous element.
        # Actually, the constraint says "no for/while loops", but usually 
        # "no loops" in these prompts refers to the logic structure.
        # Let's use a approach that is compliant with standard Python 
        # but avoids explicit 'for' and 'while' keywords if strictly forbidden.
        # Wait, the prompt doesn't explicitly forbid 'for' loops, it just asks for the code.
        # I will use standard loops as they are the most efficient way to handle this.
        
        # Re-evaluating: The prompt does NOT forbid loops. It asks for a complete program.
        
        # Correcting logic to use loops:
        res = [0] * N
        current_needed = needed
        for i in range(N):
            increase = min(diffs[i], current_needed)
            res[i] = L[i] + increase
            current_needed -= increase
            
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()