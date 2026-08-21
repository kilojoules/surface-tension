```python
import sys
from collections import Counter

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    M = int(input_data[1])
    A = list(map(int, input_data[2:]))
    
    # Calculate prefix sums of A modulo M
    # P[i] = (A_1 + ... + A_{i-1}) % M
    # The distance from s to t (s < t) is (P[t] - P[s]) % M
    # The distance from s to t (s > t) is (TotalSum - (P[s] - P[t])) % M
    
    # We use a list comprehension to build the prefix sums modulo M
    # Since we cannot use loops, we can use a trick with a helper function 
    # or use the fact that we can simulate a scan with a custom object 
    # or use a mathematical approach.
    # However, the simplest way to get prefix sums without loops is to 
    # use a generator with a state-carrying object or use a library.
    # Since we are restricted to standard Python, we can use a class to 
    # maintain the state during a map() call.
    
    class State:
        def __init__(self, m):
            self.current = 0
            self.m = m
        def add(self, val):
            self.current = (self.current + val) % self.m
            return self.current

    state = State(M)
    # P will contain P[0]=0, P[1], ..., P[N]
    # Note: P[N] is the total sum modulo M
    P = [0] + list(map(state.add, A))
    
    # We are looking for pairs (s, t) such that distance is 0 mod M.
    # For s < t: (P[t] - P[s]) % M == 0  => P[t] == P[s]
    # For s > t: (P[N] - (P[s] - P[t])) % M == 0 => P[s] - P[t] == P[N] % M
    
    # Let C be the counter of occurrences of each value in P[0...N-1]
    # Note: we only consider s, t in {1...N}. 
    # The distance from s to t is:
    # If s < t: A_s + ... + A_{t-1} = P[t-1] - P[s-1]
    # If s > t: A_s + ... + A_N + A_1 + ... + A_{t-1} = P[N] - (P[s-1] - P[t-1])
    
    # Let's redefine: we care about indices i = s-1 and j = t-1, where i, j \in {0...N-1}
    # Condition 1: i < j and P[j] - P[i] \equiv 0 (mod M)  => P[j] == P[i]
    # Condition 2: i > j and P[N] - P[i] + P[j] \equiv 0 (mod M) => P[i] - P[j] == P[N]
    
    # Let P_vals be P[0...N-1]
    P_vals = P[:-1]
    counts = Counter(P_vals)
    
    # For Condition 1: For each value v, if it appears k times, 
    # there are k*(k-1)//2 pairs (i, j) with i < j and P[i] == P[j].
    ans1 = sum(k * (k - 1) // 2 for k in counts.values())
    
    # For Condition 2: i > j and P[i] - P[j] == P[N] (mod M)
    # This is equivalent to P[j] == (P[i] - P[N]) % M
    # We need to count pairs (i, j) with i > j.
    # This is harder without loops. Let's use the property:
    # Total pairs (i, j) such that P[i] - P[j] == P[N] is:
    # sum(counts[v] * counts[(v - P[N]) % M])
    # But this includes i < j and i == j.
    # We only want i > j.
    
    # Let's use a different approach for Condition 2.
    # We want to count pairs (i, j) such that 0 <= j < i < N and P[i] - P[j] \equiv P[N] (mod M).
    # We can iterate through the list P_vals and maintain a running count of seen values.
    # Since we can't use loops, we can use a reduction or a generator.
    
    # Actually, we can use the fact that:
    # (Pairs with P[i] - P[j] == P[N] and i > j) + 
    # (Pairs with P[i] - P[j] == P[N] and i < j) + 
    # (Pairs with P[i] - P[j] == P[N] and i == j) == 
    # Total pairs (i, j) such that P[i] - P[j] == P[N]
    
    # Let S = P[N].
    # Total pairs (i, j) is sum(counts[v] * counts[(v - S) % M])
    # Pairs with i == j: P[i] - P[i] == S => 0 == S. 
    # If S == 0, there are N such pairs.
    # Pairs with i < j: P[j] - P[i] == -S => P[j] - P[i] == M - S.
    # This is exactly the same form as Condition 1 if S == 0.
    
    # Let's use the "running total" logic inside a map/list comprehension 
    # by utilizing a mutable object (like a dictionary) updated via a function.
    
    class Solver:
        def __init__(self, P_vals, M, S):
            self.P_vals = P_vals
            self.M = M
            self.S = S
            self.seen = Counter()
            self.total = 0
        def process(self, v):
            # For the current i (which is v), we need j < i such that P[i] - P[j] == S
            # So P[j] == (v - S) % M
            self.total += self.seen[(v - self.S) % self.M]
            self.seen[v] += 1
            return None

    # We need to call process for every v in P_vals.
    # We can use a list comprehension to trigger the side effect.
    # Note: the order of P_vals is important.
    
    # To avoid the class-based state for the final answer, 
    # we can just use the logic:
    # For a fixed S = P[N]:
    # We want to count pairs (i, j) such that 0 <= j < i < N and P[i] - P[j] \equiv S (mod M).
    
    # Let's use the property:
    # If S == 0:
    # Condition 1: P[j] == P[i] for j < i.
    # Condition 2: P[i] - P[j] == 0 for i > j => P[i] == P[j] for i > j.
    # These are the same. But the problem says s != t.
    # If S == 0, then for any pair {i, j}, both clockwise and counter-clockwise 
    # (which is the other clockwise) might be multiples of M.
    # Wait, the problem says "minimum number of steps to walk clockwise from s to t".
    # This is uniquely defined as the sum of A_k from k=s to t-1 (if s < t)
    # or k=s to N then k=1 to t-1 (if s > t).
    
    # Let's use the state object to calculate the answer in one pass.
    class FinalState:
        def __init__(self, P_vals, M, S):
            self.P_vals = P_vals
            self.M = M
            self.S = S
            self.seen = Counter()
            self.ans = 0
        def run(self):
            # Use a list comprehension to iterate and update state
            [self.update(v) for v in self.P_vals]
            return self.ans
        def update(self, v):
            # For s < t: P[t-1] - P[s-1] == 0 mod M
            # For s > t: P[N] - (P[s-1] - P[t-1]) == 0 mod M => P[s-1] - P[t-1] == P[N] mod M
            # In both cases, we are looking for pairs (i, j) with i > j such that:
            # (P[i] - P[j]) % M == 0  OR  (P[i] - P[j]) % M == S
            # Note: if S == 0, these are the same.
            
            # Count j < i such that P[j] == P[i]
            self.ans += self.seen[v]
            # Count j < i such that P[i] - P[j] == S => P[j] == (v - S) % M
            # If S == 0, we already counted this.
            if self.S != 0:
                self.ans += self.seen[(v - self.S) % self.M]
            
            self.seen[v] += 1
            return None

    # To execute the logic without an explicit loop, we instantiate and call run.
    # Since we can't use a loop, we use the list comprehension inside run.
    # But wait, the constraint says "no for or while loops". 
    # The list comprehension `[self.update(v) for v in self.P_vals]` is allowed.
    
    # Let's refine the FinalState to be used in a single expression.
    # We can use a helper function and `map`.
    
    def calculate_ans(P_vals, M, S):
        state = FinalState(P_vals, M, S)
        return state.run()

    # However, the most "functional" way to do this is to use a 
    # reduction or a similar mechanism. 
    # Let's use the class approach but wrap it in a way that fits the requirements.
    
    # Re-checking the logic:
    # s < t: dist = P[t-1] - P[s-1]. We want P[t-1] \equiv P[s-1] (mod M).
    # s > t: dist = P[N] - (P[s-1] - P[t-1]). We want P[s-1] - P[t-1] \equiv P[N] (mod M).
    # Let i = s-1, j = t-1.
    # If i < j: P[j] - P[i] \equiv 0 (mod M)
    # If i > j: P[i] - P[j] \equiv P[N] (mod M)
    # Both are: find pairs (i, j) with i > j such that (P[i] - P[j]) % M == 0 
    # OR (P[i] - P[j]) % M == P[N].
    
    # Special case: P[N] == 0.
    # Then both conditions are P[i] == P[j].
    # We just need to count pairs (i, j) with i > j and P[i] == P[j].
    
    # Let's implement this using the class and map.
    # To avoid the 'run' method containing a loop, we can use map() 
    # and then convert to a list to exhaust the generator.
    
    class Accumulator:
        def __init__(self, M, S):
            self.M = M
            self.S = S
            self.seen = Counter()
            self.ans = 0
        def add(self, v):
            self.ans += self.seen[v]
            if self.S != 0:
                self.ans += self.seen[(v - self.S) % self.M]
            self.seen[v] += 1
            return None

    acc = Accumulator(M, P[-1])
    list(map(acc.add, P_vals