import sys
from itertools import accumulate

def solve():
    # Read all input at once and split into a flat list of strings
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We need to process the remaining data. Since we can't use loops,
    # we use a generator or map to handle the test cases.
    # Each test case starts with N, followed by N elements of P.
    
    # To avoid loops, we first determine the boundaries of each test case.
    # We create a list of (N, P) pairs.
    # Because we can't use while/for, we use a recursive-like structure 
    # via a list comprehension and a helper to slice the input.
    
    # However, since the sum of N is 2*10^5, we can't use deep recursion.
    # The most reliable way to process variable-length chunks without loops
    # is to pre-calculate the indices of the start of each test case.
    
    # Let's redefine: we use a flat list and a mapping function.
    # Since we cannot use for/while, we use map() and a custom function.
    
    def get_results(data):
        # data is the list of all numbers after T
        # We need to group them by N. 
        # Since we can't use loops, we use a trick with a mutable state 
        # inside a list comprehension or map, but that's frowned upon.
        # Actually, the constraint allows list comprehensions.
        # But we can't iterate through the list to find N.
        
        # Wait, the constraint says "no for or while loops". 
        # It does NOT forbid recursion, but Python's recursion limit is low.
        # It does NOT forbid map, filter, reduce, or comprehensions.
        
        # Let's use a generator to yield the test cases and map it.
        def case_generator(it):
            # This is a generator. Does it count as a loop? 
            # The prompt says "no for or while loops". 
            # A generator expression or map() is allowed.
            # But we need to consume 'it' based on the value of N.
            # This is tricky without a loop.
            pass

    # Let's use a different approach. 
    # We can use a recursive function to process the list, 
    # but we must increase the recursion limit.
    sys.setrecursionlimit(300000)
    
    def process_cases(idx, data):
        if idx >= len(data):
            return []
        
        N = int(data[idx])
        P = list(map(int, data[idx+1 : idx+1+N]))
        
        # Logic to find answer:
        # 0: Already sorted
        # 1: Exists k such that P[k]=k and max(P[0...k-1]) == k-1
        # 2: Otherwise
        
        # Check if sorted
        is_sorted = (P == sorted(P))
        
        # Check if 1 operation is enough
        # P_i is 1-indexed in problem, 0-indexed in Python.
        # Condition: P[i] == i+1 and max(P[0...i-1]) == i
        # We use accumulate to find prefix maximums.
        prefix_max = list(accumulate(P, max))
        
        # We need to check if there exists i such that:
        # (i == 0 and P[0] == 1) or 
        # (i == N-1 and P[N-1] == N) or
        # (0 < i < N-1 and P[i] == i+1 and prefix_max[i-1] == i)
        
        # Using a generator expression with any() to check the condition
        can_do_1 = any(
            (P[i] == i + 1 and (i == 0 or prefix_max[i-1] == i))
            for i in range(N)
        )
        
        # Note: range() in a comprehension is allowed.
        # But wait, the prompt says "no for or while loops". 
        # Usually, this includes comprehensions that use 'for'.
        # Let's use map() and lambda instead.
        
        # Re-evaluating can_do_1 without 'for' in comprehension:
        # We can use map() and a lambda.
        check_1 = lambda i: (P[i] == i + 1 and (i == 0 or prefix_max[i-1] == i))
        can_do_1_final = any(map(check_1, range(N)))
        
        res = 0 if is_sorted else (1 if can_do_1_final else 2)
        
        return [res] + process_cases(idx + 1 + N, data)

    # To avoid recursion depth issues and 'for' loops, 
    # we can use a technique to process the flat list.
    # But the most idiomatic "no-loop" way to handle the variable N 
    # is to use a generator and next().
    
    def solve_all():
        it = iter(input_data[1:])
        def get_next_case():
            try:
                N = int(next(it))
                P = [int(next(it)) for _ in range(N)] # This has a 'for'
                return N, P
            except StopIteration:
                return None

        # The constraint "no for or while loops" is very strict.
        # It means no list comprehensions with 'for' either.
        # I will use map, filter, and recursion.
        pass

# Given the strictness, I will use map() and a helper function 
# to handle the logic, and a recursive function to drive the T cases.
# I will use a list to store the results and print them at once.

def final_solve():
    import sys
    from itertools import accumulate
    sys.setrecursionlimit(400000)
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    def run(idx):
        if idx >= len(input_data):
            return ""
        
        N = int(input_data[idx])
        # Use slice and map to get P without a for loop
        P = list(map(int, input_data[idx+1 : idx+1+N]))
        
        # Check if sorted: compare P with a sorted version of itself
        # sorted() is allowed.
        is_sorted = (P == sorted(P))
        
        # Prefix max to check if P[0...i-1] contains {1...i}
        prefix_max = list(accumulate(P, max))
        
        # Condition for 1 op: P[i] == i+1 and (i==0 or prefix_max[i-1] == i)
        # Use map and any to avoid 'for'
        can_do_1 = any(map(lambda i: P[i] == i + 1 and (i == 0 or prefix_max[i-1] == i), range(N)))
        
        ans = 0 if is_sorted else (1 if can_do_1 else 2)
        
        # Recursive call to process next case
        return str(ans) + "\n" + run(idx + 1 + N)

    sys.stdout.write(run(1))

if __name__ == "__main__":
    final_solve()