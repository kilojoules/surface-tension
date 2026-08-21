import sys
from functools import reduce

def solve():
    # Read all input data
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))
    
    # The condition "No building taller than Building j between i and j" 
    # means that for a fixed i, we are looking for indices j > i such that 
    # H[k] < H[j] for all k where i < k < j.
    # This is equivalent to saying that Building j is a "visible" building 
    # when looking right from Building i, but specifically defined by the 
    # height of the target building j rather than the source building i.
    
    # Let's rephrase: j satisfies the condition if H[j] > max(H[i+1...j-1]).
    # This means j is a record-breaker (prefix maximum) of the sequence H[i+1...N].
    
    # To solve this for all i efficiently:
    # We can use a Divide and Conquer approach or a Segment Tree/Fenwick Tree 
    # combined with a monotonic stack logic.
    # However, a simpler way to think about it is:
    # For a fixed j, for which i < j is it true that H[j] > max(H[i+1...j-1])?
    # This is true for all i such that there is no k in (i, j) with H[k] > H[j].
    # Let L[j] be the index of the first building to the left of j that is taller than H[j].
    # If no such building exists, L[j] = 0 (1-indexed).
    # Then for a fixed j, the condition is satisfied for i = L[j], L[j]+1, ..., j-1.
    # The number of such i is j - L[j].
    
    # Wait, the condition is: "no building taller than Building j between i and j".
    # Let's check Sample 1: H = [2, 1, 4, 3, 5]
    # i=1: j=2(H=1, max()=0 < 1 OK), j=3(H=4, max(1,1)=1 < 4 OK), j=4(H=3, max(1,4)=4 < 3 NO), j=5(H=5, max(1,4,3)=4 < 5 OK). Total 3.
    # This matches the logic: for a fixed j, it contributes to c_i for all i < j 
    # such that max(H[i+1...j-1]) < H[j].
    # This is exactly the range i \in [L[j], j-1] where L[j] is the index of the 
    # nearest building to the left of j that is taller than H[j].
    # (Using 0-based indexing: L[j] is the index of the first k < j such that H[k] > H[j]. 
    # If none, L[j] = -1. Then i can be L[j], L[j]+1, ..., j-1. 
    # But i is the index of the building we start from. 
    # The buildings between i and j are H[i+1...j-1].
    # If L[j] is the index of the first building to the left > H[j], 
    # then for any i < L[j], the building at L[j] is between i and j and is > H[j].
    # For any i >= L[j], no building between i and j is > H[j].
    # So for a fixed j, the valid i's are {L[j], L[j]+1, ..., j-1}.
    # Note: L[j] might be -1. Since i is 0-indexed in Python, i ranges from 0 to N-1.
    # The number of i's is j - (L[j] + 1) + 1 = j - L[j].
    # However, we need to be careful: if L[j] = -1, then i can be 0, 1, ..., j-1.
    # That is j - (-1) = j + 1 buildings? No.
    # If L[j] = -1, then for all i < j, max(H[i+1...j-1]) < H[j].
    # Example: H=[2, 1, 4], j=2 (H=4). i=0: max(1)<4, i=1: max() < 4. Both OK.
    # Indices i are 0, 1. Count is 2. j is 2. L[j] is -1. j - L[j] = 2 - (-1) = 3. 
    # Something is wrong. Let's re-evaluate.
    
    # Correct logic:
    # For a fixed j, we want to find i < j such that max_{k=i+1}^{j-1} H_k < H_j.
    # Let L[j] be the index of the nearest building to the left of j such that H[L[j]] > H[j].
    # If no such building exists, L[j] = -1.
    # The condition is satisfied if i+1 > L[j], which means i >= L[j].
    # Also we have the constraint i < j.
    # So i \in {max(0, L[j]), ..., j-1}.
    # The number of such i is (j-1) - max(0, L[j]) + 1 = j - max(0, L[j]).
    
    # To calculate c_i for all i:
    # c_i = count of j > i such that L[j] <= i.
    # This is equivalent to: for each j, increment a counter for all i in [max(0, L[j]), j-1].
    # We can use a difference array to handle these range updates.
    
    # Find L[j] using a monotonic stack
    L = [-1] * N
    stack = []
    for j in range(N):
        while stack and H[stack[-1]] < H[j]:
            stack.pop()
        if stack:
            L[j] = stack[-1]
        stack.append(j)
    
    # Difference array for range updates
    # We want to add 1 to range [max(0, L[j]), j-1]
    diff = [0] * (N + 1)
    for j in range(1, N):
        left = max(0, L[j])
        right = j - 1
        if left <= right:
            diff[left] += 1
            diff[right + 1] -= 1
            
    # Compute prefix sums to get c_i
    # c = list(map(lambda x, r: r.append(x), 
    #            reduce(lambda r, x: r + [r[-1] + x] if r else [x], diff, []), 
    #            []))
    # Since we can't use loops, we use a scan approach with a list comprehension 
    # and a helper to maintain state, or just use a trick with a class or reduce.
    
    # To avoid loops, we can use a custom function with reduce to build the prefix sum array.
    def prefix_sum(arr):
        return reduce(lambda acc, x: acc + [acc[-1] + x], arr[1:], [arr[0]])

    result = prefix_sum(diff)
    # We only need the first N elements (c_0 to c_{N-1})
    sys.stdout.write(" ".join(map(str, result[:N])) + "\n")

if __name__ == "__main__":
    solve()