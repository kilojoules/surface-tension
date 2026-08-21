import sys

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    k = int(input_data[1])

    # The total number of good sequences S is (NK)! / (K!)^N.
    # We need the floor((S+1)/2)-th sequence.
    # This is the "middle" sequence.
    # Due to the symmetry of lexicographical order, if a sequence A is the i-th,
    # its "complement" (where each value x is replaced by N - x + 1) 
    # is the (S - i + 1)-th sequence.
    # The middle sequence is the one that is "closest" to its own complement.
    # Specifically, for S sequences, the floor((S+1)/2)-th sequence is the 
    # last sequence that is lexicographically smaller than or equal to its complement.
    
    # A sequence A is lexicographically smaller than its complement A' if at the 
    # first index i where A_i != A'_i, we have A_i < A'_i.
    # A_i < A'_i means A_i < N - A_i + 1, or 2 * A_i < N + 1.
    
    # To find the floor((S+1)/2)-th sequence, we want the largest sequence A 
    # such that A <= complement(A).
    # This happens when we try to place the largest possible values at the 
    # earliest possible positions, provided the sequence remains <= its complement.
    
    # The condition A <= complement(A) is satisfied if at the first index i 
    # where A_i != (N - A_i + 1), we have A_i < (N - A_i + 1).
    # This means A_i <= floor(N/2).
    
    # To maximize A lexicographically:
    # 1. We want the first index i where A_i != A'_i to be as late as possible.
    # 2. At that index i, we want A_i to be the largest possible value that is 
    #    still less than A'_i. That value is floor(N/2).
    # 3. Before that index i, we must have A_j = A'_j, which means 
    #    A_j = N - A_j + 1, or 2 * A_j = N + 1. 
    #    This is only possible if N is odd and A_j = (N+1)/2.
    # 4. After index i, we want the sequence to be as large as possible 
    #    (descending order of available numbers).
    
    # Case 1: N is even.
    # A_j = A'_j is impossible. The first index i is 0.
    # To maximize A such that A_0 < A'_0, we set A_0 = N // 2.
    # Then we fill the rest of the sequence with the remaining numbers in 
    # descending order to get the largest possible sequence.
    
    # Case 2: N is odd.
    # We can have A_j = A'_j if A_j = (N+1) // 2.
    # We can do this for all K occurrences of (N+1)//2.
    # Then at the next index, we must have A_i < A'_i, so A_i = (N-1) // 2.
    # Then fill the rest in descending order.

    # However, the "descending order" must respect the count K.
    # The available numbers are 1...N.
    # If N is even:
    # First element: N // 2
    # Remaining: K copies of (N // 2 + 1 ... N) and K copies of (1 ... N // 2 - 1)
    # and K-1 copies of N // 2.
    # To maximize lexicographically, we output remaining numbers in descending order.
    
    # If N is odd:
    # First K elements: (N + 1) // 2
    # Next element: (N - 1) // 2
    # Remaining: K copies of ((N + 3) // 2 ... N) and K copies of (1 ... (N - 3) // 2)
    # and K-1 copies of (N - 1) // 2.
    # Output remaining in descending order.

    res = []
    if n % 2 == 0:
        mid = n // 2
        res.append(mid)
        # Remaining numbers: 
        # n...mid+1 (K times each), mid (K-1 times), mid-1...1 (K times each)
        others = sorted([x for x in range(1, n + 1) for _ in range(k)] 
                        - {mid}, reverse=True)
        # Since we can't use set subtraction on lists, we use a list comprehension
        # to get all numbers except one instance of 'mid'.
        all_nums = [x for x in range(1, n + 1) for _ in range(k)]
        # Remove one 'mid' and sort descending
        # To avoid loops, we use a generator and sorted()
        # We can't use .remove(), so we filter one instance.
        # But wait, the simplest way to get "all but one mid" sorted descending:
        # Sorted list of all, then remove the first occurrence of mid.
        # But we can't use loops. Let's use a list comprehension to 
        # create the list and then a slice to remove one.
        
        # Correct approach for "all but one mid" sorted descending:
        # 1. Create list of all NK elements.
        # 2. Sort descending.
        # 3. Remove the first occurrence of 'mid'.
        # Since we can't use loops or .remove(), we can use:
        # sorted_all = sorted(all_nums, reverse=True)
        # index = sorted_all.index(mid)
        # result = sorted_all[:index] + sorted_all[index+1:]
        
        # Let's refine this into the final output.
    else:
        mid_val = (n + 1) // 2
        res.extend([mid_val] * k)
        split_val = (n - 1) // 2
        res.append(split_val)
        # Remaining: all nums except K mid_vals and one split_val, sorted descending.
        
    # To implement the "remove one" and "sort" without loops:
    # We can use a list comprehension to build the full list and then slice.
    
    # Let's rewrite the logic to be strictly compliant.
    
    # For N even:
    # Part 1: N // 2
    # Part 2: Sorted([1...N]*K excluding one N//2), reverse=True
    
    # For N odd:
    # Part 1: (N+1)//2 repeated K times
    # Part 2: (N-1)//2
    # Part 3: Sorted([1...N]*K excluding K*(N+1)//2 and one (N-1)//2), reverse=True

    # Implementation using a helper to handle the "remove one" logic:
    # We can use a list comprehension to filter out the elements.
    # But we need to remove exactly ONE instance of a value.
    # A trick: use a list of all elements, find the index of the value, and slice.
    
    # Final logic construction:
    def get_sequence(n, k):
        if n % 2 == 0:
            mid = n // 2
            all_nums = sorted([x for x in range(1, n + 1) for _ in range(k)], reverse=True)
            idx = all_nums.index(mid)
            return [mid] + all_nums[:idx] + all_nums[idx+1:]
        else:
            mid_val = (n + 1) // 2
            split_val = (n - 1) // 2
            # All numbers except the K mid_vals
            others = [x for x in range(1, n + 1) if x != mid_val for _ in range(k)]
            all_others_sorted = sorted(others, reverse=True)
            idx = all_others_sorted.index(split_val)
            return [mid_val] * k + [split_val] + all_others_sorted[:idx] + all_others_sorted[idx+1:]

    print(*(get_sequence(n, k)))

if __name__ == "__main__":
    solve()