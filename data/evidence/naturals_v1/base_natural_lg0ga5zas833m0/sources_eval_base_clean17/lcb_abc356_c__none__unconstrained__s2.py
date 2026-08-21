import sys

def solve():
    # Read N, M, K from the first line
    try:
        line1 = sys.stdin.readline().split()
        if not line1:
            return
        n, m, k = map(int, line1)
    except ValueError:
        return

    tests = []
    for _ in range(m):
        parts = sys.stdin.readline().split()
        if not parts:
            break
        
        # The format is: C_i, then C_i keys, then R_i
        c_i = int(parts[0])
        # Keys are indices from 1 to N. We store them as 0-indexed for bitmasking.
        keys = [int(x) - 1 for x in parts[1:c_i+1]]
        result = parts[c_i+1]
        tests.append((keys, result))

    valid_combinations = 0
    
    # Since N is small (up to 15), we can iterate through all 2^N combinations.
    # A combination is represented by a bitmask where the i-th bit is 1 if key i is real.
    for mask in range(1 << n):
        is_consistent = True
        for keys, result in tests:
            # Count how many real keys are in the current test set
            real_count = 0
            for key in keys:
                if (mask >> key) & 1:
                    real_count += 1
            
            # Door opens if real_count >= K
            opened = (real_count >= k)
            
            if result == 'o':
                # Door should have opened
                if not opened:
                    is_consistent = False
                    break
            else: # result == 'x'
                # Door should not have opened
                if opened:
                    is_consistent = False
                    break
        
        if is_consistent:
            valid_combinations += 1
            
    print(valid_combinations)

if __name__ == "__main__":
    solve()