import sys

def solve():
    # Read S
    s_input = sys.stdin.readline().strip()
    if not s_input:
        return
    
    # Read Q
    q_input = sys.stdin.readline().strip()
    if not q_input:
        return
    q = int(q_input)
    
    # Read all K values
    k_values = []
    while len(k_values) < q:
        line = sys.stdin.readline().split()
        for val in line:
            k_values.append(int(val))
            
    n = len(s_input)
    results = []
    
    def flip_case(char):
        if 'a' <= char <= 'z':
            return char.upper()
        else:
            return char.lower()

    for k in k_values:
        # The process creates a sequence where the length doubles each time.
        # Let L_0 = n. After m operations, L_m = n * 2^m.
        # The character at position k (1-indexed) can be found by tracing
        # which half it falls into at each step.
        
        # We only care about the operations until the length exceeds k.
        # Since k <= 10^18 and n >= 1, we need at most 60 operations.
        
        # We need to find how many times the character at position k was 
        # part of the "T" (the flipped part) of the concatenation S + T.
        
        # current_k is 0-indexed for easier modulo arithmetic
        current_k = k - 1
        flips = 0
        
        # We trace back from the final length to the initial length n.
        # The length at step m is n * 2^m.
        # If current_k >= n * 2^(m-1), it was in the flipped part.
        # Instead of finding m, we can use the property that 
        # current_k = current_k - (length_of_S) and increment flip count.
        
        # To do this efficiently, we find the largest power of 2 such that 
        # n * 2^m <= current_k.
        
        # However, a simpler way: 
        # While current_k is outside the range [0, n-1]:
        # Find the largest L = n * 2^m such that L <= current_k.
        # This is not quite right because the structure is recursive.
        
        # Correct logic:
        # The string is S_m = S_{m-1} + T_{m-1}.
        # If k <= len(S_{m-1}), it's in S_{m-1}.
        # If k > len(S_{m-1}), it's the (k - len(S_{m-1}))-th char of T_{m-1},
        # which is the flipped version of the (k - len(S_{m-1}))-th char of S_{m-1}.
        
        temp_k = current_k
        num_flips = 0
        
        # We need to find the smallest m such that n * 2^m > temp_k.
        # Then we check if temp_k >= n * 2^(m-1).
        
        # Instead of iterating m, we can use bit manipulation if n was 1.
        # Since n is arbitrary, we find the "level" of k.
        
        # Let's find the smallest m such that n * (2^m) > temp_k.
        # This is equivalent to 2^m > temp_k / n.
        
        # Binary search for the range or just use a loop since m is small (~60).
        m = 0
        while n * (1 << m) <= temp_k:
            m += 1
            
        # Now trace back from m down to 1.
        for i in range(m, 0, -1):
            half_len = n * (1 << (i - 1))
            if temp_k >= half_len:
                num_flips += 1
                temp_k -= half_len
        
        # The final character is s_input[temp_k] flipped (num_flips % 2) times.
        char = s_input[temp_k]
        if num_flips % 2 == 1:
            char = flip_case(char)
        results.append(char)
        
    print(" ".join(results))

if __name__ == "__main__":
    solve()