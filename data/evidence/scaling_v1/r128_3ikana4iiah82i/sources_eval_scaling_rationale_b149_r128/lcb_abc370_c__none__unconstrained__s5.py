import sys

def solve():
    # Read S and T from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    s = input_data[0]
    t = input_data[1]

    # Identify indices where S and T differ
    # We need to change characters one by one.
    # To get the lexicographically smallest array X, we should prioritize
    # changing characters at the earliest possible index that results in a 
    # string smaller than other options.
    # However, the problem asks for the minimum number of elements M.
    # M is simply the number of positions where S[i] != T[i].
    # To make the array X lexicographically smallest, at each step we must
    # choose the change that results in the lexicographically smallest string.
    # This means we should process indices i from 0 to N-1, but only if
    # changing S[i] to T[i] makes the string smaller.
    # Wait, the rule is: we must reach T in M steps. 
    # In each step, we change one character.
    # To make the sequence of strings X lexicographically smallest:
    # 1. We should first change characters that make the string smaller (S[i] > T[i]).
    #    Among these, we should change the leftmost index first.
    # 2. Then we change characters that make the string larger (S[i] < T[i]).
    #    Among these, we should change the rightmost index first to keep the 
    #    prefix as small as possible for as long as possible.

    diffs = [i for i in range(len(s)) if s[i] != t[i]]
    
    # Indices that decrease the string value
    decreasers = [i for i in diffs if s[i] > t[i]]
    # Indices that increase the string value
    increasers = [i for i in diffs if s[i] < t[i]]
    
    # To minimize the array lexicographically:
    # - Handle decreasers from left to right (makes the string smaller as early as possible)
    # - Handle increasers from right to left (delays making the string larger)
    order = decreasers + increasers[::-1]
    
    # Generate the sequence of strings
    # We use a list comprehension and a helper function to apply changes
    def apply_change(current_s, index, target_t):
        return current_s[:index] + target_t[index] + current_s[index+1:]

    # Since we cannot use loops, we use a technique to accumulate the strings.
    # We can use a list comprehension with a side-effect (not recommended) 
    # or use a recursive-like structure via a comprehension.
    # Actually, the most idiomatic way without loops to generate a sequence 
    # based on previous elements is using a custom function or reduce.
    
    from functools import reduce
    
    # We want to produce the list of strings. 
    # reduce returns the final state, so we wrap the state in a list.
    def accumulate_strings(acc, index):
        current_s = acc[-1]
        return acc + [apply_change(current_s, index, t)]
    
    # Start with [s], then apply changes. The first element is the initial S,
    # but the output X should contain the strings AFTER the changes.
    result_array = reduce(accumulate_strings, order, [s])
    
    # The first element of result_array is the original S, which is not part of X.
    x = result_array[1:]
    
    # Output the results
    print(len(x))
    sys.stdout.write('\n'.join(x) + ('\n' if x else ''))

if __name__ == "__main__":
    solve()