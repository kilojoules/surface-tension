import sys
from itertools import accumulate

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Create list of (L, R) pairs
    intervals = [(int(input_data[i]), int(input_data[i+1])) 
                for i in range(1, 2 * N, 2)]
    
    # Calculate the total minimum and total maximum possible sums
    # Using map/sum to avoid explicit loops
    total_min = sum(map(lambda x: x[0], intervals))
    total_max = sum(map(lambda x: x[1], intervals))
    
    # The condition for a solution to exist is that 0 must be within [total_min, total_max]
    if total_min > 0 or total_max < 0:
        print("No")
        return

    # We need to find X_i such that sum(X_i) = 0 and L_i <= X_i <= R_i.
    # Let X_i = L_i + delta_i, where 0 <= delta_i <= R_i - L_i.
    # Then sum(L_i + delta_i) = 0  =>  sum(delta_i) = -sum(L_i).
    # Let Target = -total_min. We need to distribute Target across delta_i.
    
    target = -total_min
    # For each interval, the maximum we can add to L_i is (R_i - L_i)
    max_deltas = [r - l for l, r in intervals]
    
    # We use accumulate to keep track of the running sum of max_deltas.
    # For each i, we want to take as much as possible from max_deltas[i] 
    # without exceeding the remaining target.
    # The amount taken for index i is: 
    # min(max_deltas[i], target - sum(max_deltas[0...i-1]))
    # However, it's easier to think: we take the full max_delta unless 
    # the accumulated sum exceeds the target.
    
    acc_max = list(accumulate(max_deltas))
    
    # For each i, the contribution to the target is:
    # If acc_max[i-1] < target, we take min(max_deltas[i], target - acc_max[i-1])
    # We can use a list comprehension to build the X sequence.
    # To avoid index errors with i-1, we can prepend 0 to acc_max.
    
    shifted_acc = [0] + acc_max[:-1]
    
    # X_i = L_i + amount_taken
    # amount_taken = max(0, min(max_deltas[i], target - shifted_acc[i]))
    # But since target >= 0 and we only take if shifted_acc < target, 
    # we can simplify the logic.
    
    X = [
        l + max(0, min(r - l, target - shifted_acc[i]))
        for i, (l, r) in enumerate(intervals)
    ]
    
    # Final check: the logic above might leave a remainder if target > total_max - total_min.
    # But we already checked total_max < 0, and target is -total_min.
    # The condition for success is total_min <= 0 <= total_max.
    # If total_min <= 0, then target >= 0.
    # If total_max >= 0, then target <= sum(max_deltas).
    
    # Since we must use the exact target, and we might have "underfilled" 
    # if we didn't use a loop to update the target, let's refine:
    # The amount taken at step i is min(max_deltas[i], target - sum_of_previous_taken).
    # This is a greedy approach.
    
    # Let's redefine X using a more robust greedy:
    # We need to distribute 'target' units across N slots, each with capacity max_deltas[i].
    # The amount taken at index i is min(max_deltas[i], target - sum(taken[0...i-1])).
    
    # Since we can't use loops, we use the property that we take the full 
    # max_delta for all i < k, a partial amount for i = k, and 0 for i > k.
    # k is the index where acc_max[k] first reaches or exceeds target.
    
    # We can find k using a list comprehension and next()
    try:
        # Find the first index where the prefix sum exceeds the target
        k = next(i for i, s in enumerate(acc_max) if s >= target)
        
        # For i < k: X_i = R_i
        # For i == k: X_i = L_i + (target - acc_max[k-1])
        # For i > k: X_i = L_i
        
        # To implement this in one list comprehension:
        result = [
            r if i < k else (
                l + (target - acc_max[k-1]) if i == k else l
            )
            for i, (l, r) in enumerate(intervals)
        ]
        
        # Handle the k=0 case for acc_max[k-1]
        if k == 0:
            # target is between 0 and max_deltas[0]
            # X_0 = L_0 + target, X_i = L_i for i > 0
            result = [intervals[0][0] + target] + [l for l, r in intervals[1:]]
            
        print("Yes")
        print(*(result))
        
    except StopIteration:
        # This case is actually covered by total_max < 0, but for safety:
        print("No")

if __name__ == "__main__":
    solve()