import sys
from math import comb
from functools import reduce

def solve():
    # Read N and K from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n, k = map(int, input_data)

    # The total number of good sequences is S = (NK)! / (K!)^N
    # We need the floor((S+1)/2)-th sequence.
    # Let target = (S + 1) // 2.
    
    # To avoid calculating S explicitly (which can be massive), 
    # we can determine the sequence element by element.
    # At each position, we try numbers 1 to N.
    # If we pick number 'x', the number of ways to complete the sequence is:
    # (Remaining Total)! / Product(Remaining counts of each number!)
    
    # We use reduce to maintain a state: (current_counts, target_index, result_sequence)
    # current_counts: list of remaining counts for each number 1..N
    # target_index: the rank we are looking for
    # result_sequence: the sequence built so far
    
    initial_counts = [k] * n
    initial_target = 1 # We will calculate S and then target = (S+1)//2
    
    # First, calculate total S to find the target rank
    # S = (n*k)! / (k!)^n
    # However, we can just find the middle by comparing the target rank 
    # against the number of sequences starting with 1, 2, etc.
    
    # To find the middle sequence, we need the total count S.
    # Since we can't use loops, we use a formula for S.
    # But we can't use math.factorial for N*K = 250,000.
    # Wait, the constraints say N, K <= 500. N*K is 250,000.
    # We need a way to handle the target rank without computing S fully if possible,
    # or use Python's arbitrary precision integers.
    
    # Let's compute S using a functional approach.
    # S = comb(n*k, k) * comb((n-1)*k, k) * ... * comb(k, k)
    total_s = reduce(lambda a, b: a * b, [comb(i * k, k) for i in range(1, n + 1)], 1)
    target = (total_s + 1) // 2

    def get_ways(counts):
        # Ways to arrange remaining items: (sum(counts))! / product(counts!)
        # This is equivalent to:
        # comb(sum(counts), counts[0]) * comb(sum(counts)-counts[0], counts[1]) ...
        return reduce(lambda a, b: a * b, 
                      [comb(sum(counts[i:]), counts[i]) for i in range(n)], 1)

    def find_element(state, _):
        counts, target_rank, seq = state
        
        # We need to find which number x (1 to N) the target_rank falls into.
        # We use a helper function to iterate through 1..N and find the index.
        def find_x(current_x, current_rank):
            if current_x > n:
                return None # Should not happen
            
            # If we pick current_x, how many sequences are there?
            # We need current_x to have remaining counts > 0.
            if counts[current_x - 1] > 0:
                # Temporarily decrement count to calculate ways
                temp_counts = list(counts)
                temp_counts[current_x - 1] -= 1
                ways = get_ways(temp_counts)
                
                if current_rank <= ways:
                    return (current_x, current_rank)
                else:
                    return find_x(current_x + 1, current_rank - ways)
            else:
                return find_x(current_x + 1, current_rank)

        x, new_rank = find_x(1, target_rank)
        
        # Update counts for the next step
        new_counts = list(counts)
        new_counts[x - 1] -= 1
        
        return (new_counts, new_rank, seq + [x])

    # Use reduce to simulate the process for N*K steps
    final_state = reduce(find_element, range(n * k), (initial_counts, target, []))
    
    # Print the result sequence
    print(*(final_state[2]))

if __name__ == "__main__":
    # Increase recursion depth for find_x
    sys.setrecursionlimit(2000)
    solve()