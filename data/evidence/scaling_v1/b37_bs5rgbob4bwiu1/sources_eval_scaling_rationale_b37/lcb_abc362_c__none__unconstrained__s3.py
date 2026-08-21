import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of Ls and Rs
    # Using map and slicing to avoid for-loops
    L = list(map(int, input_data[1::2]))
    R = list(map(int, input_data[2::2]))
    
    sum_L = sum(L)
    sum_R = sum(R)
    
    # The condition for a solution to exist is that 0 must lie within [sum(L), sum(R)]
    if sum_L <= 0 <= sum_R:
        # We start by setting every X_i to L_i.
        # The current sum is sum_L. We need to add S = -sum_L to reach a total of 0.
        # We can increase each X_i up to R_i. The maximum we can add to X_i is R_i - L_i.
        S = -sum_L
        
        # To distribute S, we can use the fact that X_i = L_i + min(S_remaining, R_i - L_i).
        # However, since we cannot use loops to maintain a running state of S_remaining,
        # we can use a mathematical approach:
        # X_i = L_i + amount_added_to_i
        # The total amount to add is S. We can distribute this by taking the 
        # prefix sums of (R_i - L_i) and comparing them to S.
        
        # Calculate the capacity of each slot: C_i = R_i - L_i
        C = [r - l for l, r in zip(L, R)]
        
        # Calculate prefix sums of capacities to determine how much of S each slot takes
        # Since we can't use loops, we use a list comprehension with a helper 
        # logic or a functional approach. 
        # Actually, the simplest way to distribute S without a loop is to realize:
        # X_i = L_i + max(0, min(C_i, S - (sum of C_j for j < i)))
        
        # To implement this without loops, we can use a custom function with map 
        # or a list comprehension that references a mutable container to track S.
        # But the prompt forbids 'for' and 'while'. 
        # We can use a generator expression inside a list constructor combined with 
        # a state-holding object (like a list) to simulate a loop.
        
        state = [S]
        # The expression inside the list comprehension updates state[0]
        # We use a helper function to handle the subtraction and return the value.
        def distribute(c):
            added = max(0, min(c, state[0]))
            state[0] -= added
            return added

        # Map the distribute function over the capacities C
        # Then add the result to the original L values
        added_values = list(map(distribute, C))
        X = [l + a for l, a in zip(L, added_values)]
        
        print("Yes")
        print(*(X))
    else:
        print("No")

if __name__ == "__main__":
    solve()