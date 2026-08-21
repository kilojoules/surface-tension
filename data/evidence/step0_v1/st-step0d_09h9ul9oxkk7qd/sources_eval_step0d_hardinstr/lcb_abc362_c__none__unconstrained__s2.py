import sys

def solve():
    # Read all input at once
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    # Parse pairs into lists of L and R
    pairs = [ (int(input_data[i]), int(input_data[i+1])) for i in range(1, 2*N, 2) ]
    L = [p[0] for p in pairs]
    R = [p[1] for p in pairs]
    
    sum_L = sum(L)
    sum_R = sum(R)
    
    # Condition for existence: 0 must be within [sum(L), sum(R)]
    if sum_L > 0 or sum_R < 0:
        print("No")
        return

    # We need to increase sum_L to 0.
    needed = -sum_L
    
    # Calculate the capacity of each interval
    capacities = [r - l for l, r in zip(L, R)]
    
    # To avoid loops, we use a prefix sum of capacities to find how much 
    # of the 'needed' amount is absorbed by each element.
    # prefix_caps[i] is the sum of capacities of elements 0 to i-1.
    # We can use a list comprehension to build the prefix sum.
    # However, since N is 2*10^5, a naive O(N^2) prefix sum is too slow.
    # We use a trick: we only need to know if the 'needed' amount 
    # has been exhausted.
    
    # Since we cannot use loops, we can't use itertools.accumulate 
    # if we strictly follow "no loops" (though accumulate is a builtin).
    # Let's use a list comprehension with a helper function or 
    # a mathematical approach.
    
    # Actually, the simplest way to distribute 'needed' without a loop 
    # is to realize that for each i, the amount added is:
    # max(0, min(capacity[i], needed - sum(capacities[0...i-1])))
    
    # To get prefix sums without loops, we use itertools.accumulate.
    from itertools import accumulate
    pref = list(accumulate(capacities))
    
    # For each i, the amount added is:
    # The portion of 'needed' that falls within the range [pref[i-1], pref[i]]
    # Amount = max(0, min(pref[i], needed) - pref[i-1])
    
    # Handle pref[i-1] for i=0
    def get_added(i):
        p_current = pref[i]
        p_prev = pref[i-1] if i > 0 else 0
        return max(0, min(p_current, needed) - p_prev)
    
    X = [L[i] + get_added(i) for i in range(N)]
    
    print("Yes")
    print(*(X))

if __name__ == "__main__":
    solve()