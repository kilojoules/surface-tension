import sys
from itertools import permutations, combinations_with_replacement
from collections import Counter

def is_palindrome(s):
    return s == s[::-1]

def contains_palindrome_of_length_k(s, k):
    # Check every substring of length k
    for i in range(len(s) - k + 1):
        if is_palindrome(s[i : i + k]):
            return True
    return False

def solve():
    # Read input using sys.stdin.read().split()
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    k = int(input_data[1])
    s = input_data[2]

    # The constraints N <= 10 allow us to iterate through all unique permutations.
    # Since itertools.permutations returns all permutations based on index,
    # we can use a set to filter unique strings or use a recursive approach.
    # Given N=10, 10! is 3,628,800, which is manageable in Python if we 
    # avoid explicit Python loops where possible, but since we need to check 
    # the palindrome condition, we can use a generator expression inside sum().
    
    # To get only unique permutations without using a set (which consumes memory),
    # we can sort the string and use a custom unique permutation generator.
    # However, for N=10, the simplest way to handle duplicates is to generate 
    # all permutations and divide by the factorial of the counts of each character,
    # but that only works if the condition is independent of position.
    # Since the condition is about substrings, we must check each unique permutation.
    
    # To generate unique permutations:
    def unique_permutations(elements):
        if not elements:
            yield ()
        else:
            # Sort to ensure we can handle duplicates
            sorted_elements = sorted(elements)
            # Use a set to track which characters have been used at the current position
            # This is tricky with recursion. Let's use a simpler approach.
            pass

    # Given the constraints and the "no-loop" preference for some environments,
    # we can use a recursive function to build the string and prune/count.
    
    char_counts = Counter(s)
    chars = sorted(char_counts.keys())
    counts = [char_counts[c] for c in chars]
    
    # We use a list to keep track of the current string being built
    current_string = [None] * n
    
    def count_valid(pos):
        # Base case: string is fully constructed
        if pos == n:
            # Check if the constructed string contains a palindrome of length k
            # We convert the list to a string first
            res_str = "".join(current_string)
            return 0 if contains_palindrome_of_length_k(res_str, k) else 1
        
        # Recursive step: try each available character
        # To avoid loops, we use a list comprehension and sum
        return sum([
            count_valid(pos + 1)
            for i in range(len(chars))
            if counts[i] > 0 and (
                # Optimization: check if adding this char creates a palindrome of length k
                # The palindrome would end at index 'pos'.
                # It starts at index pos - k + 1.
                not (pos >= k - 1 and (
                    # Check if the substring from pos-k+1 to pos is a palindrome
                    # current_string[pos-k+1 ... pos-1] + chars[i]
                    # We need to check if current_string[pos-k+1 + j] == current_string[pos - j]
                    # for j from 0 to k // 2
                    all(
                        current_string[pos - k + 1 + j] == (chars[i] if j == k - 1 else current_string[pos - 1 - j])
                        for j in range(k // 2)
                    )
                ))
            )
            # We must decrement the count before recursing
            # Since we can't do that in a list comprehension easily, 
            # we use a helper to manage state.
        ])

    # Because we cannot use loops or standard recursion with state mutation easily 
    # inside a comprehension, let's use a standard recursive approach with a trick.
    
    # To strictly follow "no for/while loops" (if that's the implied challenge),
    # we can use map/filter/reduce, but recursion is allowed.
    
    def backtrack(pos, current_counts):
        if pos == n:
            return 1
        
        def try_char(i):
            char = chars[i]
            count = current_counts[i]
            if count <= 0:
                return 0
            
            # Palindrome check: if we place char at pos, does it form a palindrome of length k?
            # The palindrome would be from index (pos - k + 1) to pos.
            # We check if current_string[pos - k + 1 + j] == current_string[pos - j]
            # Note: current_string[pos] is the one we are currently placing (chars[i]).
            
            is_pal = (
                pos >= k - 1 and 
                all(
                    current_string[pos - k + 1 + j] == (chars[i] if j == k - 1 else current_string[pos - 1 - j])
                    for j in range(k // 2)
                )
            )
            
            if is_pal:
                return 0
            
            # Place character and recurse
            current_string[pos] = char
            # Create new counts tuple to avoid mutation
            new_counts = list(current_counts)
            new_counts[i] -= 1
            return backtrack(pos + 1, tuple(new_counts))

        return sum(map(try_char, range(len(chars))))

    # To avoid "RecursionError" and "for/while", we use the logic above.
    # Since we can't use for/while, we use map/sum.
    
    # We need to wrap the logic to avoid using 'for' in the outer scope.
    # The constraints on N are small (10), so recursion is fine.
    
    # To satisfy the "no for/while" constraint strictly:
    sys.setrecursionlimit(2000)
    print(backtrack(0, tuple(counts)))

if __name__ == "__main__":
    solve()