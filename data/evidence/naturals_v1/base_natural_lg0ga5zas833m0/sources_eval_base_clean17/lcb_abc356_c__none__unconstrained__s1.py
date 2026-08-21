import sys

def solve():
    # Read N, M, K from the first line
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    
    tests = []
    ptr = 3
    for _ in range(M):
        # Each test consists of:
        # C_i (number of keys), then C_i keys, then R_i (result)
        C_i = int(input_data[ptr])
        keys = []
        for j in range(C_i):
            keys.append(int(input_data[ptr + 1 + j]))
        result = input_data[ptr + 1 + C_i]
        tests.append((keys, result))
        ptr += (C_i + 2)

    valid_combinations_count = 0
    
    # There are 2^N possible combinations of real/dummy keys.
    # Since N <= 15, we can iterate through all 2^N combinations.
    # We use a bitmask where the i-th bit represents whether key (i+1) is real.
    for mask in range(1 << N):
        is_possible = True
        
        for keys, result in tests:
            # Count how many real keys are in the current test set
            real_count = 0
            for k in keys:
                # Key numbers are 1-indexed, so we check the (k-1)-th bit
                if (mask >> (k - 1)) & 1:
                    real_count += 1
            
            # Check if the result matches the condition:
            # Door opens (o) if real_count >= K
            # Door doesn't open (x) if real_count < K
            if result == 'o':
                if real_count < K:
                    is_possible = False
                    break
            else: # result == 'x'
                if real_count >= K:
                    is_possible = False
                    break
        
        if is_possible:
            valid_combinations_count += 1
            
    print(valid_combinations_count)

if __name__ == "__main__":
    solve()