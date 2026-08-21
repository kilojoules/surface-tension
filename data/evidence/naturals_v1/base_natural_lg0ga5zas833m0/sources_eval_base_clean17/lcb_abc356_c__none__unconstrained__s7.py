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
        
        # C_i is parts[0]
        # A_{i,j} are parts[1 : C_i + 1]
        # R_i is parts[C_i + 1]
        c_i = int(parts[0])
        keys = [int(x) for x in parts[1:c_i+1]]
        result = parts[c_i+1]
        tests.append((keys, result))

    valid_combinations_count = 0

    # There are 2^N possible combinations of real/dummy keys.
    # We can represent each combination as a bitmask from 0 to 2^N - 1.
    # If the j-th bit is 1, key (j+1) is real. If 0, it is a dummy.
    for mask in range(1 << n):
        is_consistent = True
        for keys, result in tests:
            # Count how many real keys are in the current test set
            real_count = 0
            for key in keys:
                # key is 1-indexed, so check (key-1)-th bit
                if (mask >> (key - 1)) & 1:
                    real_count += 1
            
            # Condition: Door opens if and only if real_count >= K
            if result == 'o':
                if real_count < k:
                    is_consistent = False
                    break
            else: # result == 'x'
                if real_count >= k:
                    is_consistent = False
                    break
        
        if is_consistent:
            valid_combinations_count += 1

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()