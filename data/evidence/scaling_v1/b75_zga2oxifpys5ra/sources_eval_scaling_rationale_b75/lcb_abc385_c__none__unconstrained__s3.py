import sys

def solve():
    # Read N and the list of heights
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We need to check every possible starting building (i)
    # and every possible interval (d) between chosen buildings.
    # For a fixed start i and interval d, we count how many 
    # buildings H[i], H[i+d], H[i+2d]... have the same height as H[i].
    # However, the condition is that ALL chosen buildings must have the same height.
    # So for a fixed i and d, we check the sequence and stop as soon as the height differs,
    # or simply filter the sequence and check if they are all identical.
    
    # Actually, the simplest way is:
    # For every pair of indices (i, j) where i < j, they define a potential 
    # height h = H[i] and an interval d = j - i.
    # We then check how many elements in the sequence i, i+d, i+2d... 
    # have the height h. Note: the problem says "chosen buildings are arranged 
    # at equal intervals", meaning we pick indices i, i+d, i+2d... 
    # and ALL of them must have the same height.
    
    # Let's use a comprehension to evaluate all possible (start, interval) pairs.
    # i: starting index (0 to N-1)
    # d: interval (1 to N-1)
    
    # For a fixed i and d, the indices are i, i+d, i+2d...
    # We count how many of these have the same height as H[i].
    # Wait, the condition is: "The chosen buildings all have the same height" 
    # AND "arranged at equal intervals".
    # This means if we pick indices {p, p+d, p+2d, ..., p+(k-1)d}, 
    # then H[p] == H[p+d] == ... == H[p+(k-1)d].
    
    # To find the maximum k:
    # For every starting position i and every interval d:
    # We can find the maximum k such that H[i] == H[i+d] == ... == H[i+(k-1)d].
    
    # Since we cannot use loops, we can use a helper function with recursion 
    # or a clever comprehension. Since N=3000, recursion depth is an issue.
    # Instead, we can iterate over all i and d, and for each, 
    # use a list comprehension to find the length of the contiguous block of identical heights.
    
    # However, the most straightforward way to count the length of the 
    # sequence H[i], H[i+d], H[i+2d]... that matches H[i] is to 
    # generate the sequence and find where it first differs.
    
    # But we can just iterate over all i and d, and for each, 
    # check the sequence H[i::d]. We want the longest prefix of H[i::d] 
    # where all elements equal H[i].
    
    # Actually, the problem doesn't say we must take a prefix. 
    # It says "choose some buildings". If we choose indices i, i+d, i+2d, 
    # then those specific buildings must have the same height.
    # It does NOT say we cannot skip buildings in the sequence.
    # "The chosen buildings are arranged at equal intervals" 
    # means the indices are p, p+d, p+2d, ..., p+(k-1)d.
    
    # So for a fixed i and d, we are looking at the sequence H[i], H[i+d], H[i+2d]...
    # We want to find a subsequence of this that consists of identical heights.
    # But the "equal intervals" applies to the indices of the original N buildings.
    # This means we pick a starting point 'i' and a step 'd', and we pick 
    # some number of elements from the resulting sequence.
    # If we pick the 1st, 3rd, and 5th elements of the sequence H[i::d], 
    # the intervals between them are 2d, 2d, which is still an equal interval.
    # Therefore, we just need to count how many elements in H[i::d] 
    # have the height H[i].
    # Wait, that's not correct. If we pick indices p, p+d, p+2d, 
    # then the interval is d. If we pick p, p+2d, p+4d, the interval is 2d.
    # So for any fixed i and d, we just need to count how many 
    # indices j = i + k*d (where j < N) satisfy H[j] == H[i].
    # NO, that is also wrong. The condition is that the CHOSEN buildings 
    # are at equal intervals. This means if we choose k buildings, 
    # their indices must be p, p+d, p+2d, ..., p+(k-1)d.
    # ALL of these must have the same height.
    
    # Correct logic:
    # For every i in 0..N-1 and every d in 1..N-1:
    # Find the maximum k such that H[i] == H[i+d] == H[i+2d] == ... == H[i+(k-1)d].
    # This is equivalent to finding the length of the prefix of H[i::d] 
    # that consists of the same value.
    
    # But wait, we can start the sequence at any i and use any d.
    # For a fixed i and d, we can't just count all H[j] == H[i] in H[i::d] 
    # because they must be consecutive terms of the arithmetic progression.
    # Actually, if we pick indices p, p+d, p+2d, then the interval is d.
    # If we pick p, p+2d, p+4d, the interval is 2d.
    # So we just need to check all p and all d, and count how many 
    # consecutive terms H[p], H[p+d], H[p+2d]... have the same height.
    
    # Since we can't use loops, we can use a recursive function to count 
    # the length of the matching prefix, but we must increase the recursion limit.
    # Or, we can use a trick: for a fixed i and d, the number of 
    # matching elements is the length of the sequence H[i::d] 
    # until the first element differs from H[i].
    
    # Let's use a helper function to calculate the length for a given i and d.
    def get_length(i, d, N, H):
        # We want the largest k such that for all 0 <= m < k, i + m*d < N 
        # and H[i + m*d] == H[i].
        # This can be solved by finding the first m where H[i + m*d] != H[i].
        # Since we can't use loops, we can use a generator and next().
        # The sequence of indices is (i + m*d for m in range((N-1-i)//d + 1))
        # We want to count how many consecutive elements match H[i].
        
        # We can use a list comprehension to create a list of booleans 
        # and then find the index of the first False.
        matches = [H[j] == H[i] for j in range(i, N, d)]
        # We need the length of the prefix of True values.
        # We can use a trick with join and split or just a custom function.
        # But we can't use loops. 
        # Let's use a recursive-like approach via a helper or a clever slice.
        # Actually, the simplest way to find the length of the leading True 
        # sequence in a list of booleans is:
        # len(matches) - (index of first False)
        # We can find the index of the first False using .index(False) 
        # if False exists in the list.
        
        try:
            return matches.index(False)
        except ValueError:
            return len(matches)

    # To avoid explicit loops, we use map/comprehensions.
    # We need to iterate i from 0 to N-1 and d from 1 to N-1.
    # The result is the maximum of get_length(i, d, N, H).
    
    # Since we need to avoid for loops, we use nested comprehensions.
    # Note: the constraint says "no for loops", but list comprehensions 
    # are explicitly allowed.
    
    ans = max([
        get_length(i, d, N, H)
        for i in range(N)
        for d in range(1, N)
    ] + [1]) # +[1] handles the case where N=1
    
    print(ans)

if __name__ == "__main__":
    solve()