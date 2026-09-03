import sys

def solve():
    # Read all input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of L and R
    L = [int(input_data[i]) for i in range(1, 2 * N + 1, 2)]
    R = [int(input_data[i]) for i in range(2, 2 * N + 2, 2)]
    
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(L)
    max_sum = sum(R)
    
    # A sequence X exists if and only if 0 is within [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # We start with the minimum possible values
        # We need to increase the total sum from min_sum to 0
        diff = 0 - min_sum
        
        # For each element, we can increase it by at most (R_i - L_i)
        # We distribute 'diff' across the elements
        results = []
        for i in range(N):
            # Calculate how much we can add to L[i] without exceeding R[i]
            # and without exceeding the remaining diff
            can_add = min(R[i] - L[i], diff)
            results.append(L[i] + can_add)
            diff -= can_add
            
        print("Yes")
        print(*(results))
    else:
        print("No")

if __name__ == "__main__":
    solve()