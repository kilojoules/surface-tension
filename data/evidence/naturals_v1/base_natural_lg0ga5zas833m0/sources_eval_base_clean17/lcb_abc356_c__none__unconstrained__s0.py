import sys

def solve():
    # Read N, M, K from the first line of input
    try:
        line = sys.stdin.readline().split()
        if not line:
            return
        n, m, k = map(int, line)
    except ValueError:
        return

    tests = []
    for _ in range(m):
        parts = sys.stdin.readline().split()
        if not parts:
            break
        
        # The format is: C_i A_{i,1} ... A_{i,C_i} R_i
        # c = int(parts[0])
        # The keys are from index 1 to c
        # The result is at index c + 1
        c = int(parts[0])
        keys = [int(x) for x in parts[1:c+1]]
        result = parts[c+1]
        tests.append((keys, result))

    valid_combinations_count = 0

    # Since N is small (up to 15), we can iterate through all 2^N combinations.
    # A combination can be represented by a bitmask where the i-th bit is 1 if key i+1 is real.
    for mask in range(1 << n):
        is_possible = True
        
        for keys, result in tests:
            # Count how many keys in this test are real in the current mask
            real_count = 0
            for key in keys:
                # key is 1-indexed, so we check the (key-1)-th bit
                if (mask >> (key - 1)) & 1:
                    real_count += 1
            
            # Check if the result matches the condition: Door opens if real_count >= K
            if result == 'o':
                if real_count < k:
                    is_possible = False
                    break
            else: # result == 'x'
                if real_count >= k:
                    is_possible = False
                    break
        
        if is_possible:
            valid_combinations_count += 1

    print(valid_combinations_count)

if __name__ == "__main__":
    solve()