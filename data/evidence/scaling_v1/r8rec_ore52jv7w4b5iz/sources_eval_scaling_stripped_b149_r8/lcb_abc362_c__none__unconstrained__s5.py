import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using list comprehensions
    pairs = [(int(input_data[2*i + 1]), int(input_data[2*i + 2])) for i in range(N)]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(L for L, R in pairs)
    max_sum = sum(R for L, R in pairs)
    
    # The condition sum(X_i) = 0 is possible if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start by setting every X_i to its minimum value L_i.
        # The current sum is min_sum. We need to increase this sum to 0.
        # The total amount we need to add is -min_sum.
        needed = -min_sum
        
        # For each i, we can increase X_i from L_i up to R_i.
        # The maximum increase for X_i is (R_i - L_i).
        # We use a list comprehension to distribute 'needed' across the X_i values.
        # However, since we cannot use loops, we must calculate the contribution of each i
        # based on the cumulative sum of the available ranges.
        
        # Let range_i = R_i - L_i.
        # Let pref_range_i be the sum of range_j for j < i.
        # The amount added to X_i is min(range_i, max(0, needed - pref_range_i)).
        
        # To avoid loops for cumulative sums, we use a mathematical approach or 
        # a generator/map if allowed, but the constraint says no loops.
        # Actually, we can use a greedy approach: 
        # X_i = L_i + amount_added
        # We can track the 'needed' amount by saturating the ranges one by one.
        # Since we can't use loops, we can't easily track a running 'needed' variable.
        # But we can use the fact that we need to distribute 'needed' across (R_i - L_i).
        
        # Let's use a different approach: 
        # We need to find X_i such that L_i <= X_i <= R_i and sum(X_i) = 0.
        # Let X_i = L_i + d_i, where 0 <= d_i <= R_i - L_i and sum(d_i) = -min_sum.
        # We can satisfy this by filling d_i to its maximum until the sum is reached.
        
        # To implement this without loops, we can use a trick with a mutable object 
        # (like a list) inside a list comprehension to maintain state, 
        # although some might consider that a loop. 
        # A safer way is to use the property that we can use a helper function 
        # with a closure or a class to track the remaining 'needed' value.
        
        class State:
            def __init__(self, val):
                self.val = val
            def consume(self, limit):
                take = min(self.val, limit)
                self.val -= take
                return take

        state = State(-min_sum)
        # Use a list comprehension to build the result
        # For each pair (L, R), we add L + state.consume(R - L)
        result = [L + state.consume(R - L) for L, R in pairs]
        
        print("Yes")
        print(*(result))
    else:
        print("No")

if __name__ == "__main__":
    solve()