import sys
from itertools import accumulate

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Pair L_i and R_i using slicing
    L = list(map(int, input_data[1::2]))
    R = list(map(int, input_data[2::2]))

    # The range of possible sums is [sum(L), sum(R)]
    # We need 0 to be within this range.
    sum_L = sum(L)
    sum_R = sum(R)

    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We need to find X_i such that sum(X_i) = 0 and L_i <= X_i <= R_i.
    # Let X_i = L_i + delta_i, where 0 <= delta_i <= (R_i - L_i).
    # We need sum(L_i + delta_i) = 0  =>  sum(delta_i) = -sum(L_i).
    # Let target = -sum(L). Since sum_L <= 0, target >= 0.
    # Also target <= sum(R) - sum(L) = sum(R - L), so it's reachable.
    
    target = -sum_L
    diffs = [r - l for l, r in zip(L, R)]
    
    # Use accumulate to find how much of the target is covered by each interval.
    # acc[i] is the sum of diffs up to index i.
    acc = list(accumulate(diffs))
    
    # For each i, the amount we can add to L_i is:
    # The portion of the target that falls within the range [acc[i-1], acc[i]].
    # We use a list comprehension to calculate delta_i for all i.
    # We handle the first element separately by treating acc[-1] as 0.
    
    # To avoid loops, we create a shifted version of acc for the lower bounds.
    acc_prev = [0] + acc[:-1]
    
    # delta_i = max(0, min(diffs[i], target - acc_prev[i]))
    # However, a simpler way: 
    # The total amount added is the intersection of [0, target] and [acc_prev, acc].
    # delta_i = max(0, min(acc[i], target) - acc_prev[i])
    # But since we know target <= sum(diffs), we can just use:
    # delta_i = max(0, min(acc[i], target) - acc_prev[i])
    
    deltas = [max(0, min(acc[i], target) - acc_prev[i]) for i in range(N)]
    
    # Final X_i = L_i + delta_i
    X = [L[i] + deltas[i] for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()