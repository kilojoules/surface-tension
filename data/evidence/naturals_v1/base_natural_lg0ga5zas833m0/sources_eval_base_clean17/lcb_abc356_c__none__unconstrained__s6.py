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
        line = sys.stdin.readline().split()
        if not line:
            break
        
        # The format is: C_i A_{i,1} ... A_{i,Ci} R_i
        # C_i is line[0], R_i is line[-1], keys are line[1:-1]
        c_i = int(line[0])
        keys = [int(x) for x in line[1:-1]]
        result = line[-1]
        tests.append((keys, result))

    valid_combinations_count = 0

    # There are 2^N possible combinations of real/dummy keys.
    # We can represent each combination as a bitmask from 0 to 2^N - 1.
    # Bit j is 1 if key (j+1) is real, 0 otherwise.
    for mask in range(1 << n):
        is_possible = True
        
        for keys, result in tests:
            # Count how many keys in the current test are real
            real_count = 0
            for key in keys:
                # Key indices are 1-based, so we check (key - 1)-th bit
                if (mask >> (key - 1)) & 1:
                    real_count += 1
            
            # Door X opens if and only if real_count >= K
            opened = (real_count >= k)
            
            if result == 'o':
                # Result 'o' means door opened. If it didn't, this mask is invalid.
                if not opened:
                    is_possible = False
                    break
            else: # result == 'x'
                # Result 'x' means door didn't open. If it did, this mask is invalid.
                if opened:
                    is_possible = False
                    break
        
        if is_possible:
            valid_combinations_count += 1

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()