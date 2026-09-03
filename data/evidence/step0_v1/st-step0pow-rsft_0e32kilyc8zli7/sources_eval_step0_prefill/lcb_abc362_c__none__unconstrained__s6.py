import sys

def solve():
    # Read all input at once for efficiency
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    l_vals = []
    r_vals = []
    
    idx = 1
    for _ in range(n):
        l_vals.append(int(input_data[idx]))
        r_vals.append(int(input_data[idx+1]))
        idx += 2
        
    # The minimum possible sum is the sum of all L_i
    # The maximum possible sum is the sum of all R_i
    min_sum = sum(l_vals)
    max_sum = sum(r_vals)
    
    # A solution exists if and only if 0 is within the range [min_sum, max_sum]
    if min_sum <= 0 <= max_sum:
        # Start with the minimum possible values
        x = list(l_vals)
        current_sum = min_sum
        
        # We need to increase the sum from min_sum to 0
        # We can increase each X_i up to R_i
        needed = -current_sum
        
        for i in range(n):
            if needed <= 0:
                break
            
            # The maximum we can add to X_i is (R_i - L_i)
            can_add = r_vals[i] - l_vals[i]
            add = min(needed, can_add)
            
            x[i] += add
            needed -= add
            
        print("Yes")
        print(*(x))
    else:
        print("No")

if __name__ == "__main__":
    solve()