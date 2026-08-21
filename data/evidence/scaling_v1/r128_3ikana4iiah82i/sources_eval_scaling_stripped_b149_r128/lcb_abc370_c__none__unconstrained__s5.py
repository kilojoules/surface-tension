import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]

    # Identify indices where S and T differ
    diff_indices = [i for i in range(len(S)) if S[i] != T[i]]
    M = len(diff_indices)

    # To minimize the array X lexicographically, we want to change characters 
    # at indices that make the resulting string as small as possible.
    # However, we must change one character per step to reach T in M steps.
    # The most "aggressive" way to make the string lexicographically small 
    # is to prioritize changing characters at the earliest possible indices 
    # to their target values in T, PROVIDED that the target value is smaller 
    # than the current value. 
    # Actually, the problem is simpler: we need to reach T in M steps.
    # In each step, we change one S[i] to T[i].
    # To make the sequence of strings X lexicographically smallest, 
    # we should pick the index i that results in the lexicographically 
    # smallest string among all available indices that still need changing.
    
    # Let's evaluate the impact of changing S[i] to T[i] for all i in diff_indices.
    # We want to pick the index i that minimizes the resulting string.
    # A string is smaller if the first differing character is smaller.
    # Therefore, we should prioritize indices i where T[i] < S[i] and i is as small as possible.
    # If we must pick an index where T[i] > S[i], we want i to be as large as possible 
    # to keep the prefix of the string unchanged for as long as possible.
    
    # Correct Strategy for Lexicographical Smallest X:
    # In each step, we have a set of indices that must be changed.
    # We want to choose index i to minimize the current string.
    # 1. If there are indices i where T[i] < S[i], the smallest such i will 
    #    decrease the string's value the most at the earliest position.
    # 2. If all remaining indices i have T[i] > S[i], the largest such i will 
    #    increase the string's value at the latest possible position.

    # Since we need to do this M times, we can pre-calculate the order.
    # Indices where T[i] < S[i] should be handled in increasing order of i.
    # Indices where T[i] > S[i] should be handled in decreasing order of i.
    
    decreasers = [i for i in diff_indices if T[i] < S[i]]
    increasers = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasers ascending, increasers descending
    order = sorted(decreasers) + sorted(increasers, reverse=True)
    
    # Generate the sequence of strings
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    x = [ "".join(s_list) for i in order if (s_list.__setitem__(i, T[i]) or True) ]
    
    # Note: the logic inside the list comprehension is a trick to mutate and return.
    # Let's rewrite it clearly.
    
    # Re-calculating X without the comprehension trick for clarity
    s_list = list(S)
    x = []
    for i in order:
        s_list[i] = T[i]
        x.append("".join(s_list))

    # Output the result
    print(M)
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()