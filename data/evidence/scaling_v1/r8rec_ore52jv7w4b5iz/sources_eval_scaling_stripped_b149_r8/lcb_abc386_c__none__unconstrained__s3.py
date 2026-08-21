import sys

def solve():
    # Read input and handle potential empty lines/extra whitespace
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    k = int(input_data[0])
    s = input_data[1]
    t = input_data[2]
    
    n, m = len(s), len(t)
    
    # The edit distance must be <= K. 
    # If length difference > K, it's impossible.
    if abs(n - m) > k:
        print("No")
        return

    # For K=1, we can check the three possible operations:
    # 1. S and T are already identical (0 operations)
    if s == t:
        print("Yes")
        return

    # 2. Replace one character: lengths must be equal
    # We check if they differ by exactly one character.
    if n == m:
        diffs = [i for i in range(n) if s[i] != t[i]]
        if len(diffs) == 1:
            print("Yes")
            return
        else:
            print("No")
            return

    # 3. Delete one character from S (or insert into T): len(s) = len(t) + 1
    if n == m + 1:
        # Find the first mismatch
        # Use a generator to find the first index where s and t differ
        first_diff = next((i for i in range(m) if s[i] != t[i]), m)
        # If we remove s[first_diff], the rest should match
        if s[first_diff + 1:] == t[first_diff:]:
            print("Yes")
        else:
            print("No")
        return

    # 4. Insert one character into S (or delete from T): len(s) + 1 = len(t)
    if n + 1 == m:
        # Find the first mismatch
        first_diff = next((i for i in range(n) if s[i] != t[i]), n)
        # If we insert a char into s at first_diff, the rest should match
        if s[first_diff:] == t[first_diff + 1:]:
            print("Yes")
        else:
            print("No")
        return

    # This part is technically unreachable given the constraints and logic above
    print("No")

if __name__ == "__main__":
    solve()