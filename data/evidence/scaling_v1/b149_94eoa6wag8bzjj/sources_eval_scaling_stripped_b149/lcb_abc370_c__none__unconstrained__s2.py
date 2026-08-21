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
    
    # To minimize the number of elements in X, we must change exactly one 
    # character per step. The minimum number of elements is the number of differences.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in the array to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest possible 
    # indices to the target characters in T, BUT only if the target character 
    # is smaller than the current character. 
    # Actually, the rule is simpler: to make the string X_i smallest, 
    # we should change the leftmost character that needs changing IF the 
    # target character is smaller than the current one. 
    # Wait, the constraint is to find the lexicographically smallest ARRAY.
    # This means we want X_1 to be as small as possible, then X_2, and so on.
    # To make X_1 smallest, we should look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    # To keep X_1 smallest, we should first process all indices i where T[i] < S[i]
    # from left to right, and then process all indices i where T[i] > S[i] 
    # from right to left (because changing a rightmost character to something 
    # larger has a smaller impact on lexicographical rank than changing a leftmost one).
    
    # Correct Strategy for lexicographically smallest array:
    # 1. Identify indices where S[i] != T[i].
    # 2. Divide these indices into two groups:
    #    Group A: indices i where T[i] < S[i] (Decreasing the character)
    #    Group B: indices i where T[i] > S[i] (Increasing the character)
    # 3. To make X_1 as small as possible, we want to decrease the leftmost 
    #    possible character. So we process Group A from left to right.
    # 4. After Group A is exhausted, we must increase characters. To keep the 
    #    string as small as possible, we increase the rightmost characters first.
    #    So we process Group B from right to left.
    
    group_a = [i for i in diff_indices if T[i] < S[i]]
    group_b = [i for i in diff_indices if T[i] > S[i]]
    
    # Order of indices to change: Group A (left to right), then Group B (right to left)
    change_order = group_a + group_b[::-1]
    
    # Generate the sequence of strings
    current_s = list(S)
    x = []
    for idx in change_order:
        current_s[idx] = T[idx]
        x.append("".join(current_s))
    
    # Output the result
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()