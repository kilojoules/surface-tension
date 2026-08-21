import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of L and R
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    sum_L = sum(L)
    sum_R = sum(R)
    
    # A sequence X exists if and only if 0 is within the range [sum_L, sum_R]
    if sum_L <= 0 <= sum_R:
        # We start with the minimum possible values
        # We need to increase the total sum from sum_L to 0.
        # The total amount we need to add is -sum_L.
        diff = -sum_L
        
        # For each X_i, we can increase it from L_i up to R_i.
        # The maximum we can add to X_i is (R_i - L_i).
        # We use a list comprehension to calculate the new X values.
        # Since we can't use for/while loops, we use a trick with a mutable 
        # state or a map, but the most straightforward way to handle the 
        # "remaining diff" without loops is using a custom function or 
        # processing the array via a method that allows state.
        
        # However, we can simply calculate X_i based on the prefix sums of (R-L).
        # Let D_i = R_i - L_i. Let S_i be the prefix sum of D.
        # X_i = L_i + (amount contributed to the total diff).
        # The contribution of index i is min(D_i, max(0, diff - S_{i-1})).
        
        # To avoid loops, we use map and a helper class to track the remaining difference.
        class State:
            def __init__(self, d):
                self.d = d
            def update(self, val):
                # val is (L_i, R_i)
                l, r = val
                contribution = min(r - l, max(0, self.d))
                self.d -= contribution
                return l + contribution

        state = State(diff)
        result = map(state.update, zip(L, R))
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()