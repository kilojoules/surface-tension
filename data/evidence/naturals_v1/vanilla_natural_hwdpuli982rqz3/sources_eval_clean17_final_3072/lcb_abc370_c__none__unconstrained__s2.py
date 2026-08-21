import sys

def solve():
    # Read S and T from standard input
    # Using read().split() to handle potential trailing newlines
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
    # For each index i where S[i] != T[i], we change S[i] to T[i].
    # However, we must consider if changing S[i] to T[i] immediately is better than 
    # changing some S[k] to T[k] where k > i.
    # Since we want the lexicographically smallest string at each step, 
    # and we can only change ONE character per step:
    # If S[i] > T[i], changing it to T[i] makes the string smaller.
    # If S[i] < T[i], changing it to T[i] makes the string larger.
    
    # To get the lexicographically smallest sequence:
    # 1. First, handle all indices i where S[i] > T[i] from left to right.
    #    (This reduces the string as quickly as possible).
    # 2. Then, handle all indices i where S[i] < T[i] from right to left.
    #    (This delays the increase of the string as long as possible).
    
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    
    # Indices where S is "too large" (decreasing them helps lexicographically)
    decrease = [i for i in diff_indices if S[i] > T[i]]
    # Indices where S is "too small" (increasing them hurts lexicographically)
    increase = [i for i in diff_indices if S[i] < T[i]]
    
    # Sort decrease indices ascending (left-to-right) to minimize the string fastest
    # Sort increase indices descending (right-to-left) to keep the prefix small longer
    order = sorted(decrease) + sorted(increase, reverse=True)
    
    # Convert S to a list because strings are immutable
    s_list = list(S)
    
    # Generate the sequence X
    # We use a list comprehension to simulate the process since we can't use for-loops
    # But we can use a helper function or a map with a side effect.
    # Given the constraints, we can just use a loop to build the result list.
    
    results = []
    # We use a trick to avoid 'for' loops if forbidden, but 'for' is not forbidden.
    # The prompt asks for a complete program.
    
    # To strictly follow the logic without mutating state in a comprehension:
    # We can use a loop to build the list.
    
    # Since I need to provide the code block:
    pass

# Re-structuring to fit all logic inside the required format
def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    N = len(S)
    
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    decrease = sorted([i for i in diff_indices if S[i] > T[i]])
    increase = sorted([i for i in diff_indices if S[i] < T[i]], reverse=True)
    order = decrease + increase
    
    s_list = list(S)
    X = []
    for i in order:
        s_list[i] = T[i]
        X.append("".join(s_list))
    
    print(len(X))
    if X:
        print("\n".join(X))

if __name__ == "__main__":
    main()