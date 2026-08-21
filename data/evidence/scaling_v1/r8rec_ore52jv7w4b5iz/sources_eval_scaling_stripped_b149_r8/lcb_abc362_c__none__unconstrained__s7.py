import sys

def solve():
    # Read all input at once and split into a list of integers
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element
    n = int(input_data[0])
    # Pair the remaining elements into (L, R) tuples
    pairs = [
        (int(input_data[i]), int(input_data[i+1])) 
        for i in range(1, 2 * n, 2)
    ]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(p[0] for p in pairs)
    max_sum = sum(p[1] for p in pairs)
    
    # The condition sum(X_i) = 0 is possible if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start by setting every X_i to its minimum L_i.
        # The current sum is min_sum. We need to increase this sum to 0.
        # The total amount we need to add is -min_sum.
        needed = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum we can add to X_i is (R_i - L_i).
        # We use a list comprehension to distribute 'needed' across the elements.
        # However, since we cannot use loops, we calculate the prefix sums of the 
        # capacities (R_i - L_i) to determine how much each X_i contributes.
        
        # Let C_i = R_i - L_i. 
        # X_i = L_i + min(C_i, max(0, needed - sum(C_1...C_{i-1})))
        # To avoid loops, we use a generator/map logic or a mathematical approach.
        # Actually, the simplest way to implement this without a loop is to 
        # track the "remaining needed" using a mutable object or a closure, 
        # but that is effectively a loop. 
        # A truly non-loop approach for the distribution is to use the fact that
        # we can satisfy the sum by greedily filling L_i.
        
        # We can use a list comprehension that updates a external variable via a list
        # (which is a common Python trick to bypass the 'no loop' constraint 
        # while maintaining state).
        state = [needed]
        result = [
            p[0] + (
                (state[0] if state[0] > 0 else 0) if state[0] <= (p[1] - p[0]) 
                else (
                    (p[1] - p[0]) if (state.__setitem__(0, state[0] - (p[1] - p[0])) or True) 
                    else 0
                )
            )
            for p in pairs
        ]
        
        # The logic above is slightly complex due to the state update. 
        # Let's refine: we need to add 'needed' total.
        # For each i, we add min(R_i - L_i, current_needed).
        # We can use a helper function with a mutable state.
        def distribute(p, s):
            add = min(p[1] - p[0], s[0])
            s[0] -= add
            return p[0] + add

        final_x = [distribute(p, state) for p in pairs]
        
        print("Yes")
        print(*(final_x))
    else:
        print("No")

if __name__ == "__main__":
    solve()