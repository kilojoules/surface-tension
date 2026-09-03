import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # To minimize the number of elements M, M must be the number of indices i where S[i] != T[i].
    # To make the array X lexicographically smallest, we want X_1 to be as small as possible, 
    # then X_2, and so on.
    # A string X_j is smaller if it has a smaller character at the earliest possible index.
    # This means we should process indices i from 0 to N-1.
    # For each index i where S[i] != T[i]:
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], changing S[i] to T[i] makes the string larger.
    # To keep the sequence lexicographically smallest, we should first handle all indices i 
    # where T[i] < S[i] (from left to right), and then handle all indices i 
    # where T[i] > S[i] (from right to left).
    # Wait, the "right to left" for T[i] > S[i] is because changing a character at index i 
    # to something larger is "bad". To delay this "badness" as much as possible 
    # for the lexicographical comparison of strings, we want the larger characters 
    # to appear as late as possible in the sequence of strings.
    # Actually, the rule is:
    # 1. Identify all i such that S[i] != T[i].
    # 2. Divide these indices into two sets: 
    #    Set A: {i | T[i] < S[i]} sorted in ascending order.
    #    Set B: {i | T[i] > S[i]} sorted in descending order.
    # 3. The sequence of indices to change is Set A followed by Set B.
    
    indices_decrease = [i for i in range(N) if T[i] < S[i]]
    indices_increase = [i for i in range(N) if T[i] > S[i]]
    
    # Sort indices_decrease ascending to make the string smaller as early as possible.
    # Sort indices_increase descending to keep the string smaller for longer.
    indices_decrease.sort()
    indices_increase.sort(reverse=True)
    
    target_indices = indices_decrease + indices_increase
    
    # Convert S to a list to allow mutations
    s_list = list(S)
    
    # Generate the sequence of strings
    result = []
    for idx in target_indices:
        s_list[idx] = T[idx]
        result.append("".join(s_list))
        
    # Output results
    print(len(result))
    for string in result:
        print(string)

if __name__ == "__main__":
    solve()