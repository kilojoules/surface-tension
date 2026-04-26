import sys

def solve():
    # Read input using sys.stdin.read().split() to avoid explicit loops
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    S_orig = input_data[0]
    Q_count = int(input_data[1])
    K_queries = map(int, input_data[2:])

    L = len(S_orig)

    def flip_case(char):
        return char.upper() if char.islower() else char.lower()

    def get_char_at(k):
        """
        The operation is S_new = S + flip(S).
        Length doubles each time.
        If k <= current_length / 2, it's the same as in the previous step.
        If k > current_length / 2, it's the flipped version of the char at (k - current_length / 2).
        The total number of flips determines the final case.
        """
        # We need to find how many times k was in the second half of the string
        # across all generations until k <= L.
        # k is 1-indexed.
        
        # Use a helper to count flips using bit manipulation.
        # The structure of the string is a Thue-Morse sequence variation.
        # The character at index (k-1) is S_orig[(k-1) % L] flipped 'popcount((k-1)//L)' times.
        
        # k_zero_indexed = k - 1
        # index_in_S = (k-1) % L
        # num_flips = bin((k-1) // L).count('1')
        # return S_orig[index_in_S] if num_flips % 2 == 0 else flip_case(S_orig[index_in_S])
        pass

    # Since we cannot use loops or recursion, we use map and a lambda.
    # The logic for the k-th character:
    # Let x = (k-1) // L. The character at index k is derived from S_orig[(k-1)%L].
    # The number of times the case is flipped is the number of 1s in the binary representation of x.
    
    process_k = lambda k: (
        lambda x, idx: (
            S_orig[idx] if bin(x).count('1') % 2 == 0 else 
            (S_orig[idx].upper() if S_orig[idx].islower() else S_orig[idx].lower())
        )
    )((k - 1) // L, (k - 1) % L)

    # Map the process_k function over all queries
    results = map(process_k, K_queries)
    
    # Join and print
    sys.stdout.write(" ".join(results) + "\n")

if __name__ == "__main__":
    solve()