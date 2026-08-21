import sys

def solve():
    # Read N from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    l = []
    r = []
    
    # Parse the L_i and R_i pairs
    idx = 1
    for _ in range(n):
        l.append(int(input_data[idx]))
        r.append(int(input_data[idx+1]))
        idx += 2
        
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(l)
    max_sum = sum(r)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # Start with the minimum possible values
        x = list(l)
        current_sum = min_sum
        
        # We need to increase the sum from min_sum to 0
        # We can increase each X_i up to R_i
        needed = -current_sum
        
        for i in range(n):
            if needed <= 0:
                break
            
            # The maximum we can add to X_i is (R_i - L_i)
            can_add = r[i] - l[i]
            add = min(can_add, needed)
            x[i] += add
            needed -= add
            
        print("Yes")
        print(*(x))
    else:
        print("No")

if __name__ == "__main__":
    solve()