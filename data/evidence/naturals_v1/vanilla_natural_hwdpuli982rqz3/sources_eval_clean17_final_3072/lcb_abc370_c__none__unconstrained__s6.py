import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # To minimize the number of elements M, we must change only the characters 
    # where S[i] != T[i]. M is the number of indices i where S[i] != T[i].
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To make the array X lexicographically smallest, we want the strings X_j 
    # to be as small as possible. 
    # A string is smaller if its earlier characters are smaller.
    # Therefore, we should prioritize changing characters at smaller indices first.
    # However, we must consider if the target character T[i] is smaller or larger than S[i].
    
    # If T[i] < S[i], changing S[i] to T[i] makes the string lexicographically smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string lexicographically larger.
    
    # Strategy for lexicographically smallest X:
    # 1. First, process all indices i where T[i] < S[i] in increasing order of i.
    #    This reduces the string value as quickly as possible.
    # 2. Then, process all indices i where T[i] > S[i] in decreasing order of i.
    #    This delays the increase of the string value as long as possible.
    
    # Note: The problem asks for the lexicographically smallest ARRAY X.
    # X_1 is the most significant. To make X_1 smallest, we want the smallest possible string.
    # If there are any i such that T[i] < S[i], picking the smallest such i and changing it 
    # to T[i] will result in a string smaller than any string produced by changing an index 
    # where T[j] > S[j].
    
    # Correct Greedy Logic:
    # To make X_1 smallest:
    # - If there are indices where T[i] < S[i], pick the smallest i and change S[i] -> T[i].
    # - If there are no indices where T[i] < S[i], but there are indices where T[i] > S[i], 
    #   pick the largest i and change S[i] -> T[i] (to keep the prefix unchanged for as long as possible).
    
    # Since we must change all diff_indices exactly once to reach T in M steps:
    # The optimal sequence of indices to change is:
    # {i | T[i] < S[i]} sorted ascending, followed by {i | T[i] > S[i]} sorted descending.
    
    part1 = [i for i in diff_indices if T[i] < S[i]]
    part2 = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort part1 ascending, part2 descending
    sequence = sorted(part1) + sorted(part2, reverse=True)
    
    # Generate the strings
    current_S = list(S)
    X = []
    for idx in sequence:
        current_S[idx] = T[idx]
        X.append("".join(current_S))
    
    # Output result
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()