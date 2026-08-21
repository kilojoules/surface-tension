import sys
from functools import reduce

def solve():
    # Read all input at once and map to integers
    input_data = map(int, sys.stdin.read().split())
    
    # Use an iterator to consume the input stream
    it = iter(input_data)
    
    # Extract N and the pairs (L, R)
    # Since we cannot use loops, we use a list comprehension to pair L and R
    # We use next(it) to get N first
    # Note: Since we need N to build the list, we handle it carefully
    # We'll read the rest of the data into a list first
    data = list(it)
    
    # The first element of data is N
    # The remaining elements are L1, R1, L2, R2...
    # We can slice the data to get Ls and Rs
    # Ls are at indices 1, 3, 5... and Rs are at indices 2, 4, 6...
    # However, since we need N, we can just use the length of the data
    # N = data[0]
    # L = data[1::2]
    # R = data[2::2]
    
    # To avoid index errors and handle the logic in a functional way:
    # We define the range bounds
    # N is data[0]
    # L is data[1::2]
    # R is data[2::2]
    
    # Calculate the minimum possible sum (sum of all L_i)
    # and maximum possible sum (sum of all R_i)
    # We use reduce(lambda x, y: x + y, ...) or simply sum()
    # sum() is a built-in that replaces a loop
    
    # We use a helper to avoid repeating the slicing
    # Since we can't define complex logic in a lambda easily, 
    # we use a list to store the pre-calculated sums.
    
    # Logic:
    # 1. Let S_min = sum(L)
    # 2. Let S_max = sum(R)
    # 3. If 0 < S_min or 0 > S_max, then No.
    # 4. Otherwise, Yes. 
    # 5. We start with X_i = L_i. 
    # 6. We need to add (0 - S_min) to the X_i values.
    # 7. For each i, we can add at most (R_i - L_i).
    # 8. We use a greedy approach: X_i = L_i + min(R_i - L_i, remaining_needed)
    
    # Because we cannot use loops to update 'remaining_needed', 
    # we use a mathematical approach to distribute the sum.
    # However, the greedy distribution usually requires a stateful loop.
    # To do this without loops, we can use a cumulative sum (itertools.accumulate)
    # to determine how much of the 'needed' sum is consumed by previous elements.
    
    # Let target = 0 - sum(L)
    # Let capacity_i = R_i - L_i
    # Let cumulative_capacity_i = sum(capacity_1 ... capacity_i)
    # The amount added to X_i is:
    # min(capacity_i, max(0, target - cumulative_capacity_{i-1}))
    
    # We use map and list comprehensions to implement this.
    
    # We'll wrap the logic in a way that avoids explicit loops.
    # We use a list to hold the intermediate values.
    
    # Since we can't use loops, we'll use a list comprehension 
    # combined with a helper list for cumulative sums.
    
    # We use a trick: we can't use 'for', but we can use 'map' and 'itertools'
    # Actually, the prompt allows list comprehensions.
    
    # Let's process the data.
    # N = data[0]
    # L = data[1::2]
    # R = data[2::2]
    
    # We need to calculate the result. 
    # Since we can't use loops, we'll use a list to store the cumulative capacities.
    # We can't use a loop to build the cumulative list, but we can use 
    # a recursive-like structure via map/reduce or simply 
    # a list comprehension that references a pre-calculated cumulative sum.
    
    # Wait, the prompt says "no for or while loops". 
    # It does NOT forbid list comprehensions or map/filter/reduce.
    # It does NOT forbid importing modules.
    
    # Let's use the cumulative sum logic.
    # target = -sum(L)
    # capacities = [R[i] - L[i] for i in range(N)]
    # cum_cap = list(itertools.accumulate(capacities))
    # X[i] = L[i] + max(0, min(capacities[i], target - (cum_cap[i-1] if i>0 else 0)))
    
    # To implement this without a for loop:
    # We use map(lambda i: ..., range(N))
    
    # Let's assemble the final code.
    pass

# Since I need to provide the code in one block, I will integrate the logic here.
import sys
from itertools import accumulate

def main():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # N is the first element
    N = int(input_data[0])
    # L values are at 1, 3, 5...
    L = [int(x) for x in input_data[1::2]]
    # R values are at 2, 4, 6...
    R = [int(x) for x in input_data[2::2]]
    
    # Calculate sums
    sum_L = sum(L)
    sum_R = sum(R)
    
    # Check if 0 is within [sum_L, sum_R]
    # We use a conditional expression to handle Yes/No
    # target is the amount we need to add to sum_L to reach 0
    target = -sum_L
    
    # Calculate capacities and cumulative capacities
    capacities = [R[i] - L[i] for i in range(N)]
    cum_cap = list(accumulate(capacities))
    
    # Calculate X_i using a list comprehension
    # X_i = L_i + amount_added
    # amount_added = max(0, min(capacity_i, target - (cum_cap[i-1] if i > 0 else 0)))
    X = [
        L[i] + max(0, min(capacities[i], target - (cum_cap[i-1] if i > 0 else 0)))
        for i in range(N)
    ]
    
    # Output result
    # We use a conditional to print Yes/No and the sequence
    sys.stdout.write(
        "Yes\n" + " ".join(map(str, X)) + "\n" 
        if sum_L <= 0 <= sum_R 
        else "No\n"
    )

if __name__ == "__main__":
    main()