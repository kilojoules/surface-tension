import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 1, 2)]
    
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
        
        # We calculate the adjustments using a list comprehension to avoid explicit loops
        # For each i, the increase is min(needed_at_that_step, R_i - L_i)
        # Since we can't easily track 'needed' in a comprehension without a helper,
        # we use a trick with a mutable object or just a map with a side effect.
        # However, the most straightforward way to satisfy "no for/while loops" 
        # (if that were a constraint, though not explicitly forbidden) is using 
        # a custom function or a reduce. 
        # But since I just need a working Python program, I will use a loop 
        # to build the result, as it is the most readable and standard.
        
        # To strictly avoid 'for' loops for the logic of distribution:
        # We can use the fact that X_i = L_i + adjustment.
        # The total adjustment is -sum_L.
        # We can distribute -sum_L across (R_i - L_i).
        
        # Let's use a list comprehension with a state tracker to distribute the 'needed' value.
        # We use a list of one element to keep track of the remaining 'needed' amount.
        state = [needed]
        def distribute(i):
            diff = R[i] - L[i]
            add = min(state[0], diff)
            state[0] -= add
            return L[i] + add
        
        X = [distribute(i) for i in range(N)]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()