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
        # Each line is: C_i A_{i,1} ... A_{i,C_i} R_i
        parts = sys.stdin.readline().split()
        if not parts:
            break
        
        c_i = int(parts[0])
        # The keys are from index 1 to c_i
        keys = [int(x) for x in parts[1:c_i+1]]
        result = parts[c_i+1]
        tests.append((keys, result))

    valid_combinations_count = 0

    # There are 2^N possible combinations of real/dummy keys.
    # We represent a combination as a bitmask where the i-th bit is 1 if key i+1 is real.
    for mask in range(1 << n):
        is_possible = True
        for keys, result in tests:
            # Count how many real keys are in the current test set
            real_count = 0
            for key in keys:
                # key is 1-indexed, so we check (key-1)-th bit
                if (mask >> (key - 1)) & 1:
                    real_count += 1
            
            if result == 'o':
                # Door opens if at least K real keys are inserted
                if real_count < k:
                    is_possible = False
                    break
            else: # result == 'x'
                # Door does not open if fewer than K real keys are inserted
                if real_count >= k:
                    is_possible = False
                    break
        
        if is_possible:
            valid_combinations_count += 1

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()