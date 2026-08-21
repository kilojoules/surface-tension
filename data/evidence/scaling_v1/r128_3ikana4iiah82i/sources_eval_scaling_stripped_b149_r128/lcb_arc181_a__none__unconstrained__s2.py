import sys
from itertools import groupby

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    ptr = 1
    
    # We need to process T test cases. Since we cannot use loops, 
    # we create a list of test case data.
    test_cases = [
        (int(input_data[ptr + i*N + 1]), input_data[ptr + i*N + 2 : ptr + (i+1)*N + 2])
        # This is tricky because N varies per test case. 
        # Let's redefine how we extract test cases.
    ]
    # Wait, the above list comprehension assumes constant N. 
    # Since N varies, I must use a different approach to group the input.
    pass

# To handle variable N without loops, we can use a generator 
# and a helper function to consume the input.
def get_cases(data):
    # data is an iterator of the input tokens
    # We use a list comprehension to build the cases.
    # However, since we can't use loops, we can't easily 
    # track the pointer. Let's use a trick with a mutable object.
    state = {"ptr": 1}
    def next_case():
        if state["ptr"] >= len(data):
            return None
        N = int(data[state["ptr"]])
        P = data[state["ptr"] + 1 : state["ptr"] + 1 + N]
        state["ptr"] += 1 + N
        return (N, P)
    
    # To avoid a loop to call next_case, we can use map/range
    # but we need to know T.
    T = int(data[0])
    return [next_case() for _ in range(T)]

# Let's refine the whole thing into a single execution block.
# Since I cannot use 'for' or 'while', I will use map, 
# list comprehensions, and a helper to manage the pointer.

def final_solve():
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # We use a list to store the current index and update it via a function
    # called inside a list comprehension.
    class State:
        def __init__(self, data):
            self.data = data
            self.ptr = 1
        def get_next(self):
            N = int(self.data[self.ptr])
            P = self.data[self.ptr + 1 : self.ptr + 1 + N]
            self.ptr += 1 + N
            return N, P

    state = State(input_data)
    
    # Process each case:
    # The operation sorts [1, k-1] and [k+1, N].
    # This means P_k remains in place, and all others are sorted.
    # If we can find a k such that P_k = k and the remaining elements 
    # (which will be sorted) fill the other slots correctly, 1 op suffices.
    # Actually, the condition for 1 op is: 
    # There exists k such that if we remove P_k, the remaining 
    # elements are exactly the set {1, ..., N} \ {k}.
    # Since P is already a permutation of 1...N, any k works 
    # as long as P_k = k.
    # If P is already sorted, 0 ops.
    # Otherwise, if there is any i where P_i = i, 1 op.
    # Otherwise, 2 ops (it's proven that 2 is the maximum needed).
    
    def calculate_ans(case):
        N, P = case
        P = [int(x) for x in P]
        # Check if already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        # Check if any P_i == i
        if any(P[i] == i + 1 for i in range(N)):
            return 1
        # Otherwise 2
        return 2

    # Use map to apply calculate_ans to the results of state.get_next()
    # called T times.
    results = map(calculate_ans, [state.get_next() for _ in range(T)])
    sys.stdout.write('\n'.join(map(str, results)) + '\n')

if __name__ == "__main__":
    final_solve()