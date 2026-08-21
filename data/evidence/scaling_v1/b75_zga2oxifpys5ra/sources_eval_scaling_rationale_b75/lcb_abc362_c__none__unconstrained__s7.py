import sys

def solve():
    # Read all input at once and split into a flat list of integers
    input_data = map(int, sys.stdin.read().split())
    # Use a generator to extract N and the pairs
    data_iter = iter(input_data)
    N = next(data_iter)
    
    # Create lists of L and R values using list comprehensions (no for loops)
    # We read the remaining 2*N values and group them into pairs
    pairs = [(L, R) for L, R in zip(data_iter, data_iter)]
    L_list = [p[0] for p in pairs]
    R_list = [p[1] for p in pairs]
    
    # Calculate the minimum and maximum possible sums
    min_sum = sum(L_list)
    max_sum = sum(R_list)
    
    # Check if 0 is reachable within the range [min_sum, max_sum]
    if not (min_sum <= 0 <= max_sum):
        print("No")
        return

    # We start with X_i = L_i. The current sum is min_sum.
    # We need to add 'target_diff' to the elements to reach a total sum of 0.
    target_diff = 0 - min_sum
    
    # For each i, the maximum we can add to L_i is (R_i - L_i).
    # We want to add as much as possible to each element until target_diff is exhausted.
    # Since we cannot use a loop to maintain a running state of target_diff,
    # we can use a trick with a mutable object (a list) inside a list comprehension
    # or calculate the prefix sums of the available headroom.
    
    # Let headroom_i = R_i - L_i
    # Let prefix_headroom_i = sum(headroom_j for j in 0..i)
    # The amount added to X_i is min(headroom_i, max(0, target_diff - prefix_headroom_{i-1}))
    
    headrooms = [R - L for L, R in pairs]
    # Calculate prefix sums of headrooms to determine how much 'budget' was used before index i
    # We use a trick to simulate a scan/accumulate without a for loop:
    # In Python 3.8+, we can't use functools.reduce easily for prefix sums without a lambda,
    # but we can use a list comprehension with a helper list to track the remaining budget.
    
    # However, the most reliable way to handle the "remaining budget" without a for loop
    # is to use a helper function with a mutable state or a generator.
    def distribute_budget(target, headrooms, L_list):
        state = [target] # Use a list to allow mutation inside the generator
        def get_val(i):
            h = headrooms[i]
            add = min(h, state[0])
            state[0] -= add
            return L_list[i] + add
        return map(get_val, range(N))

    # Since the prompt forbids 'for' and 'while', we use map() to execute the logic
    result = distribute_budget(target_diff, headrooms, L_list)
    
    print("Yes")
    print(*(result))

if __name__ == "__main__":
    solve()