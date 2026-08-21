import sys

def solve():
    # Read all input at once and split into a list of tokens
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # T is the first element
    T_str = input_data[0]
    T = int(T_str)
    
    # We use a generator to group the flat list into (N, P) pairs
    # Since we cannot use loops, we use a list comprehension to process all cases
    # The input structure is N followed by N elements of P
    
    # To handle the variable length of P for each T, we need a way to index
    # into the input_data. Since we can't use while/for, we pre-calculate 
    # the starting positions of each test case.
    
    # However, the constraint to avoid loops makes indexing tricky.
    # Let's use a recursive-like structure via map or a clever comprehension.
    # Actually, the simplest way to avoid loops while processing variable lengths
    # is to use a helper function and map it, but the helper cannot have loops.
    
    # Let's redefine: we can use a list comprehension to iterate over the range of T,
    # but the prompt says "no for or while loops". 
    # This means we must use map(), filter(), reduce(), or comprehensions.
    # Wait, comprehensions ARE loops (syntactic sugar for for-loops).
    # Usually, "no for loops" implies no `for` keyword. 
    # Let's use map() and a custom function.
    
    def process_case(args):
        # args is (N, P_list)
        N = args[0]
        P = args[1]
        
        # The goal is to find the minimum operations.
        # 0 ops: already sorted.
        # 1 op: there exists k such that sorting [1, k-1] and [k+1, N] sorts the whole array.
        # This happens if there is some k where P[k] is the only element 
        # "out of place" relative to the sorted version, or more accurately,
        # if we can pick k such that all elements P[i] for i < k are <= k
        # and all elements P[i] for i > k are >= k.
        # Actually, the condition for 1 op is:
        # There exists k such that {P_1, ..., P_{k-1}} = {1, ..., k-1} 
        # AND {P_{k+1}, ..., P_N} = {k+1, ..., N}.
        # This is equivalent to saying P_k = k and the remaining elements 
        # are partitioned correctly.
        # Wait, the operation sorts the ranges. So if we pick k, 
        # the result is sorted if and only if the set of values in P[0:k-1] 
        # is {1, ..., k-1} and the set of values in P[k:N] is {k+1, ..., N}.
        # This simplifies to: P[k-1] must be k, and the values 
        # {P_0...P_{k-2}} must be {1...k-1} and {P_k...P_{N-1}} must be {k+1...N}.
        
        # Let's find the range [L, R] that is NOT sorted.
        # L is the first index where P[i] != i+1
        # R is the last index where P[i] != i+1
        
        # To find L and R without loops:
        # Create a list of indices where P[i] != i+1
        mismatches = [i for i, v in enumerate(P) if v != i + 1]
        
        if not mismatches:
            return 0
        
        L = mismatches[0]
        R = mismatches[-1]
        
        # For 1 operation to suffice, we need to find k such that:
        # 1. k-1 is the index of the element k (P[k-1] == k)
        # 2. All elements to the left of k-1 are <= k-1 (already handled by sorting)
        # 3. All elements to the right of k-1 are >= k+1 (already handled by sorting)
        # This means the only elements that can be "out of place" must be 
        # within the ranges [0, k-2] and [k, N-1].
        # But the operation sorts those ranges. 
        # So 1 op works if there exists k such that:
        # The set of values {P_0, ..., P_{k-2}, P_k, ..., P_{N-1}} 
        # is exactly {1, ..., k-1, k+1, ..., N}.
        # This is ALWAYS true if P[k-1] == k.
        # If P[k-1] == k, then sorting the left and right parts will 
        # definitely result in (1, 2, ..., k-1, k, k+1, ..., N).
        
        # So, 1 op is possible if there is any k such that P[k-1] == k
        # AND k is NOT the only element. Wait, if P[k-1] == k, 
        # then the other N-1 elements are just a permutation of the other N-1 values.
        # Sorting them will always put them in the correct place.
        # Therefore, 1 op is possible if there exists at least one k such that P[k-1] == k.
        # BUT, the operation requires k to be the pivot.
        # If we pick k, the elements at indices 0...k-2 are sorted and k...N-1 are sorted.
        # For the whole thing to be sorted, we need P[k-1] to be k.
        
        # Is it possible that 1 op is enough even if no P[k-1] == k?
        # No, because the element at index k-1 never moves.
        # If P[k-1] != k, the final array will have P[k-1] at index k-1, so it's not sorted.
        
        # So:
        # 0 ops: already sorted.
        # 1 op: exists k such that P[k-1] == k.
        # 2 ops: otherwise. (It is proven that 2 ops are always enough for N >= 3).
        # For 2 ops: pick k=1 (sorts 2...N), then k=N (sorts 1...N-1).
        # Wait, if k=1, P becomes (P_1, 1, 2, ..., N) - no, that's not right.
        # If k=1, P[1:] is sorted. If k=N, P[:N-1] is sorted.
        # Example 3: (3, 2, 1, 7, 5, 6, 4). No P[i] == i+1.
        # k=4: P[0:3] sorted (1,2,3), P[4:7] sorted (4,5,6,7). Result: (1,2,3,7,4,5,6).
        # Then k=3: P[0:2] sorted (1,2), P[3:7] sorted (4,5,6,7). Result: (1,2,3,4,5,6,7).
        
        # Check if any P[i] == i + 1
        # We can use any() with a generator
        if any(P[i] == i + 1 for i in range(N)):
            return 1
        return 2

    # To avoid loops and recursion, we use a combination of:
    # 1. A function to split the flat list into chunks
    # 2. map() to apply the logic
    
    # Since we can't use a loop to slice the input_data, we can use a 
    # technique to group the data.
    # However, the simplest way to get N and P for each case without a loop
    # is to use a generator and `next()`. But `next()` inside a 
    # comprehension is essentially a loop.
    
    # Let's use a more robust approach:
    # We can use a helper function that processes the list and returns 
    # the result and the remaining list.
    
    def run():
        # Using a list to store the current index to simulate a pointer
        ptr = [0]
        def get_case():
            N = int(input_data[ptr[0]])
            ptr[0] += 1
            P = [int(x) for x in input_data[ptr[0] : ptr[0] + N]]
            ptr[0] += N
            return (N, P)
        
        # We need to call get_case T times. 
        # map(get_case, range(T)) will do this.
        cases = map(lambda _: get_case(), range(T))
        results = map(lambda c: process_case(c), cases)
        return "\n".join(map(str, results))

    sys.stdout.write(run() + "\n")

if __name__ == "__main__":
    solve()