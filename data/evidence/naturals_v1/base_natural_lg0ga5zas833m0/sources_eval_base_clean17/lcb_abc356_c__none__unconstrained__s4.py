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
        # c_i is parts[0], keys are parts[1:-1], result is parts[-1]
        c_i = int(parts[0])
        # Convert 1-based indexing to 0-based for bitmasking
        keys = [int(x) - 1 for x in parts[1:-1]]
        result = parts[-1]
        tests.append((keys, result))

    count = 0
    # There are 2^N possible combinations of real/dummy keys.
    # A bitmask 'mask' represents the combination: 
    # if the j-th bit is 1, key j+1 is real; if 0, it's a dummy.
    for mask in range(1 << n):
        possible = True
        for keys, result in tests:
            # Count how many real keys are in the current test set
            real_count = 0
            for key in keys:
                if (mask >> key) & 1:
                    real_count += 1
            
            if result == 'o':
                # Door opens if at least K real keys are inserted
                if real_count < k:
                    possible = False
                    break
            else: # result == 'x'
                # Door doesn't open if fewer than K real keys are inserted
                if real_count >= k:
                    possible = False
                    break
        
        if possible:
            count += 1
            
    print(count)

if __name__ == "__main__":
    solve()