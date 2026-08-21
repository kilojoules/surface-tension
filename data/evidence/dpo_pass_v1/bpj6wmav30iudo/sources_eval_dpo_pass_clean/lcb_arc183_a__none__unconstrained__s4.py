import sys
from functools import reduce
from math import comb

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N, K = map(int, input_data)

    # The total number of good sequences S is (N*K)! / (K!)^N
    # We want the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    
    # To avoid calculating S (which is massive), we use the property of symmetry.
    # The S-th sequence is the reverse-complement of the 1st.
    # The target sequence is the one where we try to pick the smallest possible 
    # digits, but since we are at the halfway point, we start by picking 
    # digits such that we skip exactly (S-1)//2 sequences.
    
    # Total permutations of multiset: (sum(counts))! / product(counts!)
    # We can calculate this using combinations: 
    # comb(total, c1) * comb(total-c1, c2) * ...
    
    # We use reduce to build the sequence step by step.
    # state: (current_counts, target_rank)
    # current_counts: list of remaining counts for each number 1..N
    
    initial_counts = tuple([K] * N)
    
    # Total S = (N*K)! / (K!)^N
    # We need rank = (S + 1) // 2
    # Since S can be huge, we use Python's arbitrary precision integers.
    
    # Precompute S using a functional approach
    total_s = reduce(lambda acc, i: acc * comb(i, K), range(K, N * K + 1, K), 1) 
    # Note: The above is a simplification. Correct S:
    # S = comb(NK, K) * comb(NK-K, K) * ... * comb(K, K)
    
    # Correct S calculation using reduce
    s_val = reduce(lambda acc, i: acc * comb(i, K), range(N*K, K-1, -K), 1)
    target_rank = (s_val + 1) // 2

    def get_next_element(state):
        counts, rank = state
        total_rem = sum(counts)
        
        # Find the smallest digit d (1 to N) such that 
        # the number of sequences starting with digits < d is less than rank,
        # and sequences starting with digits <= d is >= rank.
        
        # For a chosen digit d, the number of ways to arrange the remaining is:
        # (total_rem - 1)! / (product of counts! where count[d-1] is decremented)
        # This equals: (total_rem - 1 choose count[d-1]-1) * (remaining choose K)...
        # Simplified: Ways(counts) * (count[d-1] / total_rem)
        
        # We find d by iterating through 1..N. 
        # Since we can't use loops, we use a list comprehension and next().
        
        # To calculate ways for a specific d:
        # ways = (total_rem - 1)! / ( (counts[0])! ... (counts[d-1]-1)! ... (counts[N-1])! )
        # ways = [ (total_rem - 1)! / (product counts!) ] * counts[d-1]
        # The term in bracket is (Total_Ways_of_Current_State) / total_rem
        
        current_total_ways = reduce(lambda acc, i: acc * comb(i, counts[i-1]), 
                                    range(total_rem, 0, -1), 1) 
        # Wait, the logic above for current_total_ways is slightly wrong.
        # Correct ways to arrange remaining after picking d:
        # W(d) = (total_rem - 1)! / (K!^N / counts[d-1]) 
        # But counts change.
        
        # Let's use a simpler approach for W(d):
        # W(d) = comb(total_rem - 1, counts[0]) * ... * comb(total_rem - 1 - sum(counts[:d-1]), counts[d-1]-1) ...
        
        # Actually, W(d) = (total_rem - 1)! / ( (counts[0])! ... (counts[d-1]-1)! ... )
        # W(d) = [ (total_rem - 1)! / (counts[0]! ... counts[N-1]!) ] * counts[d-1]
        # Let Base = (total_rem - 1)! / (product counts!)
        # W(d) = Base * counts[d-1]
        
        # To avoidを recalculating factorials, we observe:
        # Total ways for current state S_curr = (total_rem)! / (product counts!)
        # W(d) = S_curr * counts[d-1] / total_rem
        
        s_curr = reduce(lambda acc, i: acc * comb(i, counts[i-1]), range(total_rem, 0, -1), 1)
        # This is still a bit off. Let's use the most reliable multiset permutation formula:
        # S_curr = (sum counts)! / product(counts!)
        
        # Since we can't use loops, we'll use a helper to calculate multiset perms
        # using reduce and comb.
        
        # Let's redefine the state transition inside reduce.
        return None

    # Because of the complexity of rank-based selection and the "no loop" constraint,
    # I will use a list comprehension to generate the sequence by 
    # calculating the rank offset at each position.
    
    # We use a list to store counts and update it. Since we can't mutate in a loop,
    # we use reduce to pass the counts and the current rank.
    
    def step(state, _):
        counts, rank = state
        total_rem = sum(counts)
        
        # Calculate S_curr: total permutations of the current multiset
        # S_curr = comb(total_rem, counts[0]) * comb(total_rem-counts[0], counts[1]) ...
        s_curr = reduce(lambda acc, i: acc * comb(total_rem - sum(counts[:i]), counts[i]), range(N), 1)
        
        # For each digit d from 1 to N, the number of sequences starting with d is:
        # W(d) = S_curr * counts[d-1] / total_rem
        
        # We find d such that sum(W(1)...W(d-1)) < rank <= sum(W(1)...W(d))
        # Since we can't loop, we use a list comprehension to find the digit.
        
        # Calculate W(d) for all d
        # W = [ (s_curr * c) // total_rem for c in counts ]
        # We need the smallest d where sum(W[:d]) >= rank.
        
        # To avoid a loop to find d, we use a list comprehension with next()
        # But we need the counts for the next step.
        
        # We can't easily use next() inside reduce without a loop-like structure.
        # However, we can use a list comprehension to find the index.
        
        # Let',s calculate the cumulative sums of W
        # W_d = (s_curr * counts[d-1]) // total_rem
        # We find d such that:
        # rank > W(1) -> d=2, rank = rank - W(1)
        # rank > W(2) -> d=3, rank = rank - W(2)...
        
        # Since we can't loop, we use a mathematical trick or a very long expression.
        # We can find d by:
        # d = 1 + (number of d' where sum(W[:d]) < rank)
        
        # This is still tricky without mutation. 
        # Let's use a list comprehension to evaluate the digit and the new rank.
        
        # We can use a recursive-like structure with reduce by passing the 
        # current rank and counts.
        
        # To find d:
        # We can use a list comprehension to calculate the prefix sums of W.
        # W_prefix = [sum(W[:i]) for i in range(N+1)]
        # d = (index of first element in W_prefix >= rank)
        
        # Let's implement this logic inside the reduce:
        
        # 1. Calculate W for all d
        # 2. Find d
        # 3. Update counts and rank
        
        # Since we need to return (new_counts, new_rank, digit), 
        # we'll store the sequence in a list.
        
        return None

    # To strictly adhere to "no loops", we use reduce to iterate over the length NK.
    # The state is (counts, rank, sequence)
    
    result = reduce(
        lambda state, _: (
            (lambda counts, rank: (
                (lambda total_rem, s_curr: (
                    (lambda W: (
                        (lambda d: (
                            # New counts: decrement count of digit d
                            # New rank: rank - sum(W[:d-1])
                            # Digit: d
                            (
                                tuple([counts[i] - (1 if i == d-1 else 0) for i in range(N)]),
                                rank - sum(W[:d-1]),
                                d
                            )
                        ))(
                            # Find d: the first index where cumulative W >= rank
                            # Using a list comprehension and index()
                            [i + 1 for i in range(N) if sum(W[:i+1]) >= rank][0]
                        )
                    ))(
                        # W[d-1] = (s_curr * counts[d-1]) // total_rem
                        [(s_curr * c) // total_rem for c in counts]
                    )
                ))(
                    sum(counts),
                    # s_curr = (sum counts)! / product(counts!)
                    reduce(lambda acc, i: acc * comb(sum(counts) - sum(counts[:i]), counts[i]), range(N), 1)
                )
            ))(state[0], state[1])
        ),
        range(N * K),
        (initial_counts, target_rank)
    )
    # The above reduce returns only the last element's (counts, rank, digit).
    # We need the sequence. We can use a list to accumulate.
    # But we can't mutate. We can return the sequence in the state.
    
    # Corrected reduce to keep track of the sequence:
    final_state = reduce(
        lambda state, _: (
            (lambda counts, rank, seq: (
                (lambda total_rem, s_curr: (
                    (lambda W: (
                        (lambda d: (
                            tuple([counts[i] - (1 if i == d-1 else 0) for i in range(N)]),
                            rank - sum(W[:d-1]),
                            seq + [d]
                        ))(
                            [i + 1 for i in range(N) if sum(W[:i+1]) >= rank][0]
                        )
                    ))(
                        [(s_curr * c) // total_rem for c in counts]
                    )
                ))(
                    sum(counts),
                    reduce(lambda acc, i: acc * comb(sum(counts) - sum(counts[:i]), counts[i]), range(N), 1)
                )
            ))(state[0], state[1], state[2])
        ),
        range(N * K),
        (initial_counts, target_rank, [])
    )
    
    print(*(final_state[2]))

solve()