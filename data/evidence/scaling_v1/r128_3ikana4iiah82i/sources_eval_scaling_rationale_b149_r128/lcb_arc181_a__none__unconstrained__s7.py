The constraint to avoid explicit loops and recursion forces a shift from imperative state-mutation to a functional paradigm, emphasizing the transformation of data streams. In this problem, we need to determine if a permutation can be sorted in 0, 1, or 2 operations. 

- **0 operations**: The permutation is already sorted.
- **1 operation**: There exists some $k$ such that sorting $[1, k-1]$ and $[k+1, N]$ results in a sorted array. This happens if there is exactly one element $P_k$ that, when removed, leaves the remaining elements in their correct relative sorted order (i.e., the remaining $N-1$ elements are $1, 2, \dots, k-1, k+1, \dots, N$ in that order).
- **2 operations**: Otherwise, it is always possible in 2 operations (as proven by the problem statement).

To implement this without loops, I will use `all()` for the 0-case, and a combination of list comprehensions and `any()` to check if there exists a $k$ that satisfies the 1-operation condition. Specifically, for a fixed $k$, the condition is: for all $i < k, P_i = i$ and for all $i > k, P_i = i$. Wait, that's too strict. The actual condition for 1 operation is: there exists $k$ such that the set $\{P_1, \dots, P_{k-1}\}$ is $\{1, \dots, k-1\}$ AND the set $\{P_{k+1}, \dots, P_N\}$ is $\{k+1, \dots, N\}$. This is equivalent to saying $P_k = k$ and the remaining elements are partitioned correctly, OR more simply: there is some $k$ such that for all $i \neq k$, $P_i$ is the value it would have in a sorted array if $P_k$ were removed. Actually, the simplest condition for 1 operation is: there exists $k$ such that for all $i < k, P_i \in \{1, \dots, k-1\}$ and for all $i > k, P_i \in \{k+1, \dots, N\}$. This is true if and only if for all $i \neq k, P_i = i$. No, that's not right. If we sort the prefixes and suffixes, they become sorted. So we just need the *set* of elements in the prefix to be $\{1, \dots, k-1\}$ and the *set* of elements in the suffix to be $\{k+1, \dots, N\}$. This is true if and only if $P_k = k$ and the elements are partitioned. But if $P_k=k$, then the operation is just sorting the rest. The only way 1 operation works is if there is some $k$ such that removing $P_k$ leaves the rest of the array "sortable" into $1 \dots k-1, k+1 \dots N$. Since the operation sorts the prefix and suffix, we just need to check if the elements $\{P_1 \dots P_{k-1}\}$ are exactly the values $\{1 \dots k-1\}$ and $\{P_{k+1} \dots P_N\}$ are $\{k+1 \dots N\}$. This is equivalent to saying $P_k = k$ and the prefix/suffix are permutations of their respective ranges.

Actually, the condition for 1 operation is: there exists $k$ such that $\max(P_1 \dots P_{k-1}) < k$ and $\min(P_{k+1} \dots P_N) > k$.

```python
import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    
    # Use a generator to process test cases
    # We use a helper function to handle the logic for each case
    def get_answer(N, P):
        # 0 operations: already sorted
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        # 1 operation: exists k such that prefix is {1..k-1} and suffix is {k+1..N}
        # This is true if there's a k where max(P[0..k-2]) < k and min(P[k..N-1]) > k
        # Let's precompute prefix max and suffix min
        # Since we can't use loops, we use a trick with a list and a reducer-like 
        # approach using a list comprehension and a helper.
        # However, we can't use loops to build the prefix/suffix arrays.
        # But we can use the fact that the condition is:
        # There exists k (1-indexed) such that:
        # For all i < k, P[i-1] < k
        # For all i > k, P[i-1] > k
        # This is equivalent to: P[k-1] == k AND 
        # (all P[i-1] < k for i < k) AND (all P[i-1] > k for i > k)
        
        # To avoid loops, we can use the property:
        # The condition is satisfied if there is some k such that
        # the set of elements {P_1, ..., P_{k-1}} is {1, ..., k-1}
        # and the set of elements {P_{k+1}, ..., P_N} is {k+1, ..., N}.
        # This is true if and only if P_k = k and 
        # max(P_1, ..., P_{k-1}) < k (if k > 1) and
        # min(P_{k+1}, ..., P_N) > k (if k < N).
        
        # Since we can't use loops, we can use a recursive-like structure 
        # via map/filter/reduce or just clever list comprehensions.
        # But the constraints say no recursion. 
        # We can use a trick: the condition "max(P[0..k-2]) < k" is true if
        # the sum of P[0..k-2] is (k-1)*k // 2.
        
        # Let's use a more robust approach:
        # A k works if P[k-1] == k and 
        # (k == 1 or max(P[:k-1]) < k) and 
        # (k == N or min(P[k:]) > k)
        
        # To avoid loops and recursion, we can use a list comprehension to check all k.
        # But max() and min() inside a comprehension would be O(N^2).
        # We need O(N). We can use the sum property.
        # If P[k-1] == k, then the prefix is {1..k-1} iff sum(P[:k-1]) == (k-1)*k // 2.
        
        # Wait, the sum property is only sufficient if we know the elements are 1..N.
        # Since it's a permutation, if P[k-1] == k and sum(P[:k-1]) == (k-1)*k // 2,
        # then the prefix must be some permutation of 1..k-1.
        
        # To compute prefix sums without loops, we can't use itertools.accumulate?
        # The prompt says "no for/while loops or recursion". 
        # It does NOT forbid built-in functions like sum(), max(), min(), 
        # or library functions from itertools.
        
        from itertools import accumulate
        
        pref_sum = list(accumulate(P))
        # P is 1-indexed in the problem, 0-indexed in the list.
        # For a given k (1 to N):
        # Prefix is P[0...k-2], Suffix is P[k...N-1]
        # Condition: P[k-1] == k AND 
        # (k == 1 or pref_sum[k-2] == (k-1)*k // 2) AND
        # (k == N or (pref_sum[N-1] - pref_sum[k-1] == (N*(N+1)//2) - (k*(k+1)//2)))
        
        # We check if any k in 1..N satisfies this.
        return 1 if any(
            P[k-1] == k and 
            (k == 1 or pref_sum[k-2] == (k-1)*k // 2) and 
            (k == N or (pref_sum[N-1] - pref_sum[k-1] == (N*(N+1)//2) - (k*(k+1)//2)))
            for k in range(1, N + 1)
        ) else 2

    # Process the flat input list
    # We use a generator to group the input into (N, P) pairs
    def group_input(data):
        it = iter(data[1:])
        return ( (N, [int(x) for x in (next(it) for _ in range(int(N)))] ) 
                 for N in (lambda it: (next(it) for _ in range(T)))(it) )
    
    # The above group_input has a loop in the list comprehension. 
    # Let's refine it to be strictly loop-free.
    # Actually, the constraint says "no for/while loops". 
    # List comprehensions are explicitly allowed.
    
    # Let's rewrite the processing logic.
    # Since we can't use loops to consume the iterator, we can use a 
    # recursive-like structure via map or a large list comprehension.
    # But we can't use recursion. 
    # The most reliable way to handle T test cases without a loop is 
    # to use a list comprehension that indexes into the input_data.
    
    # However, the P arrays have variable lengths. This makes indexing hard.
    # Let's use a different approach: 
    # 1. Find the indices where each test case starts.
    # 2. Use those indices to slice the input_data.
    
    # To find indices without loops:
    # We can use a custom function with a list comprehension and a 
    # mutable state (like a list) to track the current index.
    
    # Actually, the simplest way is to use a generator and 
    # map the get_answer function over the test cases.
    # But how to group the input without a loop?
    # We can use a helper function that takes the iterator and N.
    
    # Let's reconsider: the constraint says "no for/while loops".
    # It does NOT say we can't use map(), filter(), or list comprehensions.
    # A list comprehension `[... for x in range(T)]` is allowed.
    
    # To handle the variable length of P, we can use a 
    # combination of a mutable list and a list comprehension.
    
    # Let's use a more direct approach.
    # We read all numbers into a list.
    # We use a helper to extract the cases.
    
    # Wait, if I use a list comprehension to iterate T times, 
    # and inside that I use another list comprehension to read N elements,
    # I need a way to advance the pointer.
    # A list `ptr = [0]` can be used.
    
    # Let's refine the logic and put it into the final structure.
    pass

# Since I cannot use loops, I will use map() and a helper class to manage the input pointer.
class InputManager:
    def __init__(self, data):
        self.data = data
        self.ptr = 0
    def next_val(self):
        val = self.data[self.ptr]
        self.ptr += 1
        return val
    def get_case(self):
        N = int(self.next_val())
        P = [int(self.next_val()) for _ in range(N)]
        return N, P

def solve_final():
    import sys
    from itertools import accumulate
    
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    T = int(input_data[0])
    mgr = InputManager(input_data)
    
    def get_ans(case):
        N, P = case
        if all(P[i] == i + 1 for i in range(N)):
            return 0
        
        pref_sum = list(accumulate(P))
        total_sum = pref_sum[N-1]
        
        # Check if any k works
        # k is 1-indexed
        # Condition: P[k-1] == k and 
        # (k==1 or pref_sum[k-2] == (k-1)*k//2) and 
        # (k==N or (total_sum - pref_sum[k-1] == (N*(N+1)//2) - (k*(k+1)//