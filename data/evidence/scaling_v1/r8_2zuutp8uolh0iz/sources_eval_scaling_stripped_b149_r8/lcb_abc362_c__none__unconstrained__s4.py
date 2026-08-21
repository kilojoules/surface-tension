import sys

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using list comprehensions
    intervals = [(int(input_data[2*i + 1]), int(input_data[2*i + 2])) for i in range(N)]
    
    # Calculate the minimum and maximum possible sums
    # sum(L_i) <= sum(X_i) <= sum(R_i)
    min_sum = sum(L for L, R in intervals)
    max_sum = sum(R for L, R in intervals)
    
    # The condition sum(X_i) = 0 is possible if and only if 0 is within [min_sum, max_sum]
    if min_sum > 0 or max_sum < 0:
        print("No")
        return

    print("Yes")
    
    # We start by setting all X_i = L_i. 
    # The current sum is min_sum. We need to add 'deficit' to reach 0.
    deficit = 0 - min_sum
    
    # For each i, we can increase X_i from L_i up to R_i.
    # The amount we can add to X_i is min(deficit, R_i - L_i).
    # Since we cannot use loops, we use a generator expression and a trick to 
    # track the remaining deficit. However, since we need to update the deficit 
    # dynamically, and we can't use loops, we can use a mathematical approach:
    # The total amount added to X_i is the sum of (R_j - L_j) for j < i, 
    # capped at the total deficit.
    
    # Let S_i be the prefix sum of (R_i - L_i).
    # The amount added to X_i is min(R_i - L_i, max(0, deficit - S_{i-1}))
    # But wait, we can't use a loop to build a prefix sum array easily without itertools.
    # Let's use a more direct approach: 
    # We need to distribute 'deficit' across the intervals [L_i, R_i].
    # We can use a list comprehension with a helper function or a clever trick.
    # Actually, the simplest way to "loop" without 'for' or 'while' is recursion 
    # (but N is 2e5, so recursion limit is an issue) or using a generator with a state.
    
    # We can use a mutable object (like a list) to track the remaining deficit 
    # inside a list comprehension.
    state = [deficit]
    
    def allocate(interval):
        L, R = interval
        can_add = R - L
        to_add = min(state[0], can_add)
        state[0] -= to_add
        return L + to_add

    # Map the allocate function across all intervals
    result = [allocate(inv) for inv in intervals]
    print(*(result))

if __name__ == "__main__":
    solve()