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
    # character per step. The minimum M is the number of differing characters.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in the array to be lexicographically smaller.
    # This means we should prioritize changing characters at the earliest 
    # possible indices to the target characters in T, BUT only if the 
    # target character T[i] is smaller than the current character S[i].
    # Actually, the rule is simpler: to make the resulting string X_i 
    # lexicographically smallest, we should process indices i from 0 to N-1.
    # If we change S[i] to T[i], and T[i] < S[i], the string becomes smaller.
    # If T[i] > S[i], the string becomes larger.
    # To keep X_1, X_2... smallest, we should first perform all changes 
    # where T[i] < S[i] (in increasing order of i), and then all changes 
    # where T[i] > S[i] (in increasing order of i).
    
    # Wait, the constraint is to find the lexicographically smallest ARRAY.
    # X_1 must be as small as possible. To make X_1 small, we should change 
    # the first index i where S[i] != T[i] if T[i] < S[i].
    # If for the first diff index i, T[i] > S[i], we cannot make the string 
    # smaller by changing that index. However, we might make it smaller by 
    # changing a later index j where T[j] < S[j].
    # But changing a later index j will always result in a string 
    # lexicographically larger than changing the earliest possible index i 
    # if T[i] < S[i].
    
    # Correct Strategy for lexicographically smallest array X:
    # 1. Identify all indices i where S[i] != T[i].
    # 2. Divide these indices into two sets:
    #    - Set A: indices i where T[i] < S[i] (Decreasing the character)
    #    - Set B: indices i where T[i] > S[i] (Increasing the character)
    # 3. To make X_1 as small as possible, we should first exhaust all 
    #    changes in Set A in increasing order of index.
    # 4. Then, we perform all changes in Set B in increasing order of index.
    
    set_a = [i for i in diff_indices if T[i] < S[i]]
    set_b = [i for i in diff_indices if T[i] > S[i]]
    
    # The sequence of indices to change
    change_order = set_a + set_b
    
    # Generate the array X
    # We use a list of characters for S to allow mutation
    s_list = list(S)
    x = []
    for idx in change_order:
        s_list[idx] = T[idx]
        x.append("".join(s_list))
        
    # Output the result
    print(len(x))
    for string in x:
        print(string)

if __name__ == "__main__":
    solve()