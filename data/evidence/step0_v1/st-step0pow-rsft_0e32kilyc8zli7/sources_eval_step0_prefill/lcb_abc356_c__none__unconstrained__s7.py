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
        # C_i is line[0], A_{i,j} are line[1:-1], R_i is line[-1]
        c_i = int(line[0])
        keys = [int(x) for x in line[1:-1]]
        result = line[-1]
        tests.append((keys, result))

    # There are 2^N possible combinations of real/dummy keys.
    # We can represent each combination as a bitmask from 0 to 2^N - 1.
    # Bit j is 1 if key (j+1) is real, 0 otherwise.
    
    valid_combinations_count = 0
    
    for mask in range(1 << n):
        is_consistent = True
        for keys, result in tests:
            # Count how many keys in this test are real according to the current mask
            real_count = 0
            for key in keys:
                if (mask >> (key - 1)) & 1:
                    real_count += 1
            
            # Door opens if real_count >= K
            opened = (real_count >= k)
            
            if result == 'o':
                if not opened:
                    is_consistent = False
                    break
            else: # result == 'x'
                if opened:
                    is_consistent = False
                    break
        
        if is_consistent:
            valid_combinations_count += 1
            
    print(valid_combinations_count)

if __name__ == "__main__":
    solve()