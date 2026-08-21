import sys

def solve():
    # Read input strings S and T
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    S = input_data[0]
    T = input_data[1]
    
    N = len(S)
    
    # Identify indices where S and T differ
    diff_indices = [i for i in range(N) if S[i] != T[i]]
    M = len(diff_indices)
    
    # To get the lexicographically smallest array X, we want the strings 
    # appearing earlier in X to be lexicographically smaller.
    # This means we should prioritize changing characters at the beginning of the string
    # to their target values in T if the target is smaller than the current,
    # or handle them in a specific order to ensure the resulting string is minimized.
    
    # Specifically, if we change S[i] to T[i]:
    # 1. If T[i] < S[i], doing this as early as possible makes the string smaller.
    # 2. If T[i] > S[i], doing this as late as possible keeps the string smaller.
    
    # Correct Strategy for Lexicographical Minimum Array:
    # First, process all indices i where T[i] < S[i] in increasing order of i.
    # Then, process all indices i where T[i] > S[i] in decreasing order of i.
    
    # However, the constraint is simply to reach T in M steps.
    # To make X_1 smallest, we want the smallest possible character at the earliest possible position.
    # If we can change S[i] to T[i] where T[i] < S[i], we should do it immediately for the smallest i.
    # If we must change S[i] to T[i] where T[i] > S[i], we should delay it as much as possible.
    
    # Let's refine:
    # We have a set of indices {diff_indices}. We must pick one index per step.
    # To make X_1 minimum:
    # Find the first index i in diff_indices. 
    # If T[i] < S[i], changing S[i] -> T[i] immediately is optimal.
    # If T[i] > S[i], we should check if there is any j > i such that T[j] < S[j].
    # If such j exists, changing S[j] first is better because it doesn't affect the prefix up to i,
    # and we avoid increasing S[i] until we absolutely have to.
    
    # Actually, the simplest optimal strategy:
    # 1. Change all S[i] to T[i] where T[i] < S[i] in increasing order of i.
    # 2. Change all S[i] to T[i] where T[i] > S[i] in decreasing order of i.
    
    # Wait, let's re-evaluate. 
    # Example: S = "ba", T = "ab"
    # Option 1: S[0]->'a' (aa), S[1]->'b' (ab). X = ("aa", "ab")
    # Option 2: S[1]->'b' (ba), S[0]->'a' (ab). X = ("ba", "ab")
    # "aa" < "ba", so Option 1 is better.
    
    # Example: S = "ab", T = "ba"
    # Option 1: S[0]->'b' (bb), S[1]->'a' (ba). X = ("bb", "ba")
    # Option 2: S[1]->'a' (aa), S[0]->'b' (ba). X = ("aa", "ba")
    # "aa" < "bb", so Option 2 is better.
    
    # General Rule:
    # To make the current string smallest, we want to decrease characters at the front 
    # and increase characters at the back.
    
    decreasing = [i for i in diff_indices if T[i] < S[i]]
    increasing = [i for i in diff_indices if T[i] > S[i]]
    
    # Sort decreasing indices ascending (front to back)
    # Sort increasing indices descending (back to front)
    order = sorted(decreasing) + sorted(increasing, reverse=True)
    
    # Generate the sequence of strings
    current_S = list(S)
    X = []
    for i in order:
        current_S[i] = T[i]
        X.append("".join(current_S))
        
    # Output results
    print(M)
    for string in X:
        print(string)

if __name__ == "__main__":
    solve()