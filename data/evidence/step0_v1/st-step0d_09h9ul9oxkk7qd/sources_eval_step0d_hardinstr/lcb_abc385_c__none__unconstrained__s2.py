import sys
from itertools import product

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))

    # We need to check every possible starting building (i) 
    # and every possible interval (d).
    # i ranges from 0 to n-1
    # d ranges from 1 to n-1
    
    # For a fixed start i and interval d, the buildings are at indices:
    # i, i+d, i+2d, ... as long as index < n.
    # They must all have the same height as building i.
    
    # We use a generator expression to calculate the count for all (i, d) pairs.
    # The count is the number of k such that i + k*d < n and h[i + k*d] == h[i].
    # However, the condition "equal intervals" implies we check a sequence.
    # If we pick buildings at i, i+d, i+2d... the condition is satisfied 
    # if we only pick the ones that match the height. 
    # WAIT: The problem says "The chosen buildings are arranged at equal intervals."
    # This means if we choose buildings at indices p1, p2, ..., pk, 
    # then p2-p1 = p3-p2 = ... = d.
    # This implies we are looking for the longest arithmetic progression of indices
    # where all corresponding buildings have the same height.
    
    # For a fixed start i and interval d, we can pick buildings at i, i+d, i+2d...
    # But we can only pick them if they have the same height.
    # If we encounter a building with a different height, we cannot just skip it
    # and keep the interval d, because the resulting set must be at equal intervals.
    # Actually, the condition is: we choose a set of indices {i, i+d, i+2d, ..., i+(k-1)d}.
    # All these must have height H_i.
    
    # To find the maximum k for a fixed i and d:
    # We need to find the largest k such that h[i] == h[i+d] == h[i+2d] ... == h[i+(k-1)d].
    # This is slightly different from just counting. We need the contiguous sequence.
    
    # However, the problem can be interpreted as: pick any d, and any starting point i,
    # then count how many indices in the sequence i, i+d, i+2d... have height H_i.
    # "The chosen buildings are arranged at equal intervals" means the indices 
    # are p, p+d, p+2d... 
    # Let's re-read: "The chosen buildings are arranged at equal intervals."
    # This means if you choose k buildings, their indices must be a, a+d, a+2d, ..., a+(k-1)d.
    
    # For a fixed i and d, we want to find the maximum k such that 
    # h[i] == h[i+d] == h[i+2d] == ... == h[i+(k-1)d].
    # But we can't use loops. We can use a helper function with recursion or 
    # a clever combination of map/filter.
    
    # Actually, the simplest way to check a specific (i, d) is to 
    # check the sequence and find the first index that fails.
    # Since N is small (3000), we can't do O(N^3). O(N^2) is required.
    
    # Let's refine: For every pair (i, j) where i < j and h[i] == h[j],
    # they could be the first and second elements of a sequence with d = j - i.
    # But we need to find the length of the sequence.
    
    # Correct approach:
    # For every possible interval d (1 to N), and every starting position i (0 to d-1),
    # we have a sequence h[i], h[i+d], h[i+2d]...
    # In this sequence, we look for the longest run of identical values.
    
    # Since we can't use loops, we can use groupby from itertools.
    from itertools import groupby
    
    # For a fixed d, we split the list into d groups.
    # For each group, we find the max length of identical consecutive elements.
    
    # We can use a list comprehension to iterate over d, and inside, 
    # another to iterate over the groups.
    
    # To avoid loops, we use map/max/sum.
    
    # For a fixed d:
    # We create slices h[i::d] for i in 0...d-1.
    # For each slice, we use groupby to find lengths of identical blocks.
    
    # The result is max(length of block) across all d, i.
    
    # Using a nested comprehension:
    # max(
    #   max(
    #     [len(list(g)) for k, g in groupby(h[i::d])]
    #     for i in range(d)
    #   )
    #   for d in range(1, n)
    # )
    
    # We must handle the case where N=1 separately or ensure the range is safe.
    # Also, the 1-building case is always possible.
    
    # To implement this without any 'for' loops (even in comprehensions), 
    # we use map, reduce, and other functionals. 
    # Wait, the prompt says "avoid explicit loops", usually meaning 'for' and 'while' keywords.
    # Comprehensions are generally accepted as functional constructs, but to be safe,
    # I will use map() and max().
    
    # Let's use a helper function for the groupby logic.
    def get_max_run(slice_h):
        return max([len(list(g)) for k, g in groupby(slice_h)] + [0])

    # We need to iterate d from 1 to n-1.
    # For each d, we iterate i from 0 to d-1.
    
    # Using map() to replace the loops:
    # result = max(map(lambda d: max(map(lambda i: get_max_run(h[i::d]), range(d))), range(1, n)))
    
    # Since we can't use 'for' in comprehensions, we use map/lambda.
    # But the prompt says "avoid explicit loops", and usually, 
    # comprehensions are the "Pythonic" way to do this. 
    # However, to be strictly loop-free:
    
    ans = max(map(
        lambda d: max(
            map(
                lambda i: max([len(list(g)) for k, g in groupby(h[i::d])] + [0]),
                range(d)
            ),
            default=0
        ),
        range(1, n)
    ), default=1)
    
    # The above still has a list comprehension inside max(). 
    # Let's replace [len(list(g)) for k, g in groupby(h[i::d])] 
    # with map(lambda x: len(x[1]), groupby(h[i::d]))
    
    # Final attempt at a fully loop-free expression:
    # result = max(map(lambda d: max(map(lambda i: max(map(lambda g: len(list(g[1])), groupby(h[i::d])), default=0), range(d)), default=0), range(1, n)), default=1)
    
    # Let's wrap it in a function and print.
    print(ans)

# To strictly follow "no for loops", I will define the logic using map/max.
# I will use a helper to avoid deep nesting.

def solve_final():
    input_data = sys.stdin.read().split()
    if not input_data: return
    n = int(input_data[0])
    h = list(map(int, input_data[1:]))
    if n == 0:
        print(0)
        return

    from itertools import groupby
    
    # Logic: 
    # For each interval d in 1..n-1:
    #   For each start i in 0..d-1:
    #     Get the sequence h[i], h[i+d], ...
    #     Find the longest run of identical elements.
    
    # We use map() instead of for-loops.
    # We use a list comprehension only if absolutely necessary, 
    # but map(len, ...) is better.
    
    res = max(map(
        lambda d: max(
            map(
                lambda i: max(
                    map(lambda g: len(list(g[1])), groupby(h[i::d])),
                    default=0
                ),
                range(d)
            ),
            default=0
        ),
        range(1, n)
    ), default=1)
    
    print(res)

if __name__ == "__main__":
    solve_final()