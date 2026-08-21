import sys
from itertools import product

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

    # We need to change characters at diff_indices one by one.
    # To make the array X lexicographically smallest, we want the strings 
    # appearing earlier in X to be lexicographically smaller.
    # This means we should prioritize changing characters at earlier indices 
    # to their target values in T, BUT only if the target character is 
    # smaller than the current character. 
    # Actually, the rule is: we must change one character per step.
    # To keep X_1 smallest, we want to change the leftmost possible character 
    # to the smallest possible value that helps us reach T.
    # Since we must reach T in M steps, each step must fix one incorrect character.
    # To make X_1 smallest, we look for the first index i where S[i] != T[i].
    # If T[i] < S[i], changing S[i] to T[i] immediately makes the string smaller.
    # If T[i] > S[i], we want to delay this change as much as possible to keep 
    # the string smaller for longer.
    
    # Correct Strategy for Lexicographical Smallest X:
    # 1. Identify all indices i where S[i] != T[i].
    # 2. We must change these indices one by one.
    # 3. To keep X_j small, we want to perform changes that decrease the 
    #    string's value as early as possible, and changes that increase it 
    #    as late as possible.
    # 4. Specifically:
    #    - Changes where T[i] < S[i] should be done as early as possible, 
    #      ordered by index i (left to right).
    #    - Changes where T[i] > S[i] should be done as late as possible, 
    #      ordered by index i (right to left).
    
    # Let's refine: 
    # We have a set of indices to change. In each step, we pick one index.
    # To make X_1 smallest:
    # - If there are indices i where T[i] < S[i], picking the smallest such i 
    #   and changing S[i] to T[i] is the best move.
    # - If there are no indices where T[i] < S[i], we must pick an index where 
    #   T[i] > S[i]. To keep the string smallest, we should pick the largest 
    #   such i (the rightmost one) to keep the prefix unchanged.
    
    # This looks like a greedy approach. Let's simulate it.
    # However, the number of indices is small (up to 100), but the number of 
    # permutations is huge. The greedy logic:
    # While there are indices to fix:
    #   1. Find all i where S[i] != T[i] and T[i] < S[i].
    #      If exists, pick the minimum such i.
    #   2. Otherwise, find all i where S[i] != T[i] and T[i] > S[i].
    #      Pick the maximum such i.
    
    # Let's trace Sample 1: S=adbe, T=bcbc. Diff indices: 0(a->b), 1(d->c), 3(e->c)
    # Step 1: T[1]='c' < S[1]='d' and T[3]='c' < S[3]='e'. 
    # Min index is 1. S becomes acbe. X_1 = acbe.
    # Step 2: T[3]='c' < S[3]='e'. Min index is 3. S becomes acbc. X_2 = acbc.
    # Step 3: T[0]='b' > S[0]='a'. Only one left. S becomes bcbc. X_3 = bcbc.
    # Result: 3, acbe, acbc, bcbc. Matches sample!
    
    # Implementation of the greedy simulation:
    current_S = list(S)
    target_T = list(T)
    
    # We can't use a while loop with mutations in a functional way easily, 
    # but we can use a recursive-like structure or a list comprehension 
    # if we track the state. Since we need to output the sequence, 
    # a simple loop is allowed as long as it's contained in the function.
    
    # To avoid 'while', we can use a generator or a trick with 'reduce'.
    # But the prompt allows a complete Python program. 
    # I will use a list comprehension with a helper function or just a loop.
    
    def get_next_state(s, t):
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if not diffs: return s, None
        
        # Priority 1: T[i] < S[i], pick smallest i
        decreases = [i for i in diffs if t[i] < s[i]]
        if decreases:
            idx = min(decreases)
        else:
            # Priority 2: T[i] > S[i], pick largest i
            idx = max(diffs)
            
        new_s = list(s)
        new_s[idx] = t[idx]
        return "".join(new_s), idx

    # Since I need to avoid 'while', I'll use a list comprehension to 
    # drive the state changes by calling a function that updates a global-ish 
    # state or using a recursive-like approach.
    # Actually, the simplest way to implement the greedy logic without 'while' 
    # is to use a recursive function with a decorator for memoization or 
    # just a standard recursive function.
    
    def generate_x(s, t):
        diffs = [i for i in range(len(s)) if s[i] != t[i]]
        if not diffs:
            return []
        
        decreases = [i for i in diffs if t[i] < s[i]]
        idx = min(decreases) if decreases else max(diffs)
        
        new_s = list(s)
        new_s[idx] = t[idx]
        next_s = "".join(new_s)
        
        return [next_s] + generate_x(next_s, t)

    result_x = generate_x(S, T)
    
    # Output
    print(len(result_x))
    for x in result_x:
        print(x)

if __name__ == "__main__":
    solve()