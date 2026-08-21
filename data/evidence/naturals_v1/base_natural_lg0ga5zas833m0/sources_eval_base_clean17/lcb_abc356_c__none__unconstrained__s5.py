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
        
        # The format is: C_i, then C_i keys, then R_i
        c_i = int(line[0])
        keys = [int(x) for x in line[1:c_i+1]]
        result = line[c_i+1]
        
        # We store keys as a bitmask for efficiency
        # Key j is represented by the (j-1)-th bit
        mask = 0
        for key in keys:
            mask |= (1 << (key - 1))
        
        tests.append((mask, result))

    count = 0
    # There are 2^N possible combinations of real keys.
    # We iterate through all possible bitmasks where a set bit represents a real key.
    for i in range(1 << n):
        possible = True
        for mask, result in tests:
            # Count how many real keys are in the current test set
            # i & mask extracts the keys present in the test that are real
            real_keys_in_test = bin(i & mask).count('1')
            
            if result == 'o':
                # Door opens if at least K real keys are inserted
                if real_keys_in_test < k:
                    possible = False
                    break
            else: # result == 'x'
                # Door does not open if fewer than K real keys are inserted
                if real_keys_in_test >= k:
                    possible = False
                    break
        
        if possible:
            count += 1

    print(count)

if __name__ == "__main__":
    solve()