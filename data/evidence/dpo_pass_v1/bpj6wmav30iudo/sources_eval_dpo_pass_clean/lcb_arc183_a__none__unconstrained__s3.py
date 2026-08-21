import sys
from functools import reduce
from math import factorial

def solve():
    # Read N and K from stdin
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # Total number of good sequences S = (N*K)! / (K!^N)
    # Target rank is (S + 1) // 2
    # We use reduce to build the sequence step by step.
    # state: (current_rank, current_counts) 
    # where current_counts is a tuple of remaining counts for each number 1..N
    
    def get_total_perms(counts):
        # Multinomial coefficient: sum(counts)! / product(counts!)
        return factorial(sum(counts)) // reduce(lambda a, b: a * factorial(b), counts, 1)

    # We need to find the sequence. Since N, K <= 500, S is massive.
    # We use Python's arbitrary precision integers.
    
    # Initial state
    initial_counts = tuple([K] * N)
    total_S = get_total_perms(initial_counts)
    target_rank = (total_S + 1) // 2

    # Use reduce to determine the sequence elements one by one.
    # The accumulator is (rank, counts, sequence)
    # We iterate over the range of the total length NK.
    
    result = reduce(
        lambda state, _: (
            # Calculate which number i (1..N) fits at this position
            # We find the smallest i such that the sum of perms for 1..i-1 < rank
            # and the sum of perms for 1..i >= rank.
            (
                # New rank: rank - (perms of numbers smaller than chosen i)
                state[0] - sum(
                    [get_total_perms(
                        tuple(state[1][j] - (1 if j == k else 0) for j in range(N))
                        for k in range(chosen_i - 1)
                    ]) if (chosen_i := next(
                        (i for i in range(1, N + 1) if state[1][i-1] > 0 and 
                         sum([get_total_perms(tuple(state[1][j] - (1 if j == k else 0) for j in range(N))) 
                              for k in range(i-1) if state[1][k] > 0]) < state[0] 
                              and sum([get_total_perms(tuple(state[1][j] - (1 if j == k else 0) for j in range(N))) 
                                      for k in range(i) if state[1][k] > 0]) >= state[0]])
                        or N)) # This logic is recursive in nature, handled by the comprehension
                , 0]),
                # New counts: decrement the chosen number
                tuple(state[1][j] - (1 if j == (next(
                    (i-1 for i in range(1, N + 1) if state[1][i-1] > 0 and 
                     sum([get_total_perms(tuple(state[1][m] - (1 if m == k else 0) for m in range(N))) 
                          for k in range(i-1) if state[1][k] > 0]) < state[0] 
                     and sum([get_total_perms(tuple(state[1][m] - (1 if m == k else 0) for m in range(N))) 
                              for k in range(i) if state[1][k] > 0]) >= state[0])) or N-1) 
                    for j in range(Nen := N)]),
                # Append chosen number to sequence
                state[2] + [next(
                    (i for i in range(1, N + 1) if state[1][i-1] > 0 and 
                     sum([get_total_perms(tuple(state[1][m] - (1 if m == k else 0) for m in range(N))) 
                          for k in range(i-1) if state[1][k] > 0]) < state[0] 
                     and sum([get_total_perms(tuple(state[1][m] - (1 if m == k else 0) for m in range(N))) 
                              for k in range(i) if state[1][k] > 0]) >= state[0])) or N]
            )
        ),
        range(N * K),
        (target_rank, initial_counts, [])
    )

    # The logic above is complex for a single reduce. 
    # Let's use a simpler approach: the target rank is (S+1)//2.
    # For the first element, we check if rank <= perms(1, ...). If so, first element is 1.
    # If rank > perms(1, ...), we subtract perms(1, ...) and check element 2.
    
    # To strictly avoid loops/recursion, we can use a helper function 
    # and map/reduce, but the logic for "finding the i-th element" 
    # is essentially a reduction over the length of the sequence.
    
    # Since the prompt requires NO loops/recursion, I will use a 
    # list comprehension to pre-calculate the sequence by 
    # maintaining state in a way that doesn't violate the rules.
    # However, since state depends on the previous element, 
    # reduce is the only way.
    
    # Correcting the logic into a clean reduce:
    def find_digit(rank, counts):
        # Find the smallest digit d such that the sum of permutations 
        # of prefixes smaller than d is less than rank.
        # We use a list comprehension to calculate the counts for each possible digit.
        perms_per_digit = [
            get_total_perms(tuple(counts[j] - (1 if j == d else 0) for j in range(N)))
            if counts[d] > 0 else 0
            for d in range(N)
        ]
        # Find the digit index where the cumulative sum reaches the rank
        # We use a generator to find the first index that satisfies the condition
        return next(d for d, p in enumerate(perms_per_digit) 
                    if rank <= sum(perms_per_digit[:d]) + p)

    # Final attempt at a loop-free structure using reduce
    final_state = reduce(
        lambda state, _: (
            # Calculate digit
            (lambda d: (
                # New rank: rank - sum of perms of digits < d
                state[0] - sum([get_total_perms(tuple(state[1][j] - (1 if j == k else 0) for j in range(N))) 
                               for k in range(d) if state[1][k] > 0]),
                # New counts: decrement digit d
                tuple(state[1][j] - (1 if j == d else 0) for j in range(N)),
                # Append digit (d+1) to sequence
                state[2] + [d + 1]
            ))(find_digit(state[0], state[1]))
        ),
        range(N * K),
        (target_rank, initial_counts, [])
    )
    
    print(*(final_state[2]))

if __name__ == "__main__":
    solve()