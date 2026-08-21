import sys
from itertools import accumulate

def solve():
    # Read input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Create list of (L_i, R_i) pairs
    intervals = [(int(input_data[2*i + 1]), int(input_data[2*i + 2])) for i in range(N)]
    
    # Calculate the minimum and maximum possible sums of X_i
    # min_sum = sum(L_i), max_sum = sum(R_i)
    # We use a generator expression inside sum()
    min_sum = sum(L for L, R in intervals)
    max_sum = sum(R for L, R in intervals)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We need to find X_i such that L_i <= X_i <= R_i and sum(X_i) = 0.
        # Let X_i = L_i + d_i, where 0 <= d_i <= (R_i - L_i).
        # Then sum(L_i + d_i) = 0  =>  sum(d_i) = -sum(L_i).
        # Let target = -min_sum. We need to distribute 'target' across d_i.
        target = -min_sum
        
        # For each interval, the maximum we can add to L_i is (R_i - L_i).
        # We take the minimum of the remaining target and the available room (R_i - L_i).
        # accumulate is used to track the total distributed amount to calculate the remaining target.
        
        # Calculate the "room" available for each X_i
        rooms = [R - L for L, R in intervals]
        
        # We need to find d_i such that sum(d_i) = target and 0 <= d_i <= rooms[i].
        # We can greedily fill d_i.
        # The amount added to X_i is min(rooms[i], target - sum(d_1...d_{i-1})).
        
        # To avoid loops, we use a mathematical approach to find d_i:
        # The total room is max_sum - min_sum.
        # We need to fill 'target' amount of that room.
        # We can use a prefix sum of rooms to determine how much of the target is consumed.
        
        prefix_rooms = list(accumulate(rooms))
        total_room = prefix_rooms[-1]
        
        # For each i, the amount d_i is:
        # d_i = min(rooms[i], max(0, target - (prefix_rooms[i-1] if i>0 else 0)))
        # However, since we know target <= total_room and target >= 0,
        # we can simply calculate the intersection of the range [prefix_rooms[i-1], prefix_rooms[i]]
        # and the range [0, target].
        
        # The length of the intersection of [a, b] and [0, target] is:
        # max(0, min(b, target) - max(a, 0))
        
        # Let's build the result list X
        # X_i = L_i + max(0, min(prefix_rooms[i], target) - (prefix_rooms[i-1] if i > 0 else 0))
        
        # Using a list comprehension to build the final sequence
        res = [
            intervals[i][0] + max(0, min(prefix_rooms[i], target) - (prefix_rooms[i-1] if i > 0 else 0))
            for i in range(N)
        ]
        
        print("Yes")
        print(*(res))
    else:
        print("No")

if __name__ == "__main__":
    solve()