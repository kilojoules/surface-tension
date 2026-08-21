import sys
from itertools import product

def solve():
    # Read input from standard input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # We want to find the maximum count of buildings with the same height H[i]
    # located at indices i, i+d, i+2d, ...
    # i is the starting index (0 to N-1)
    # d is the interval (1 to N-1)
    
    # For a fixed start i and interval d, the indices are range(i, N, d)
    # We count how many indices j in that range have H[j] == H[i]
    # However, the condition "arranged at equal intervals" implies we check 
    # the sequence i, i+d, i+2d... and we can only pick those that match H[i].
    # BUT, the problem asks for the maximum number of buildings we can CHOOSE.
    # If we choose buildings at indices (i, i+d, i+2d...), they are at equal intervals.
    # To satisfy the height condition, all chosen buildings must have height H[i].
    # Thus, for a fixed i and d, we count how many j in {i, i+d, i+2d...} have H[j] == H[i].
    
    # To maximize this, we iterate through all possible starts i and intervals d.
    # Since N <= 3000, a naive O(N^3) might be slow, but we only check d where 
    # the number of elements (N-i)//d is potentially greater than the current max.
    
    # Using list comprehension to evaluate all combinations:
    # For each i and d, we calculate the count of buildings with height H[i] 
    # at positions i, i+d, i+2d...
    # Note: The "equal intervals" means the distance between any two adjacent 
    # chosen buildings is the same. This is exactly what range(i, N, d) provides.
    
    # To optimize and avoid O(N^3), we observe that for a fixed i and d, 
    # we only care about the count of H[j] == H[i] for j in range(i, N, d).
    
    # We use map and max to find the result.
    # We only need to check d from 1 to N.
    # result = max( count(i, d) for i in 0..N-1 for d in 1..N )
    
    # Since we cannot use loops, we use product and a generator expression.
    # We use a conditional to handle the case where N=1.
    
    ans = max([
        sum(1 for j in range(i, N, d) if H[j] == H[i])
        for i, d in product(range(N), range(1, N))
    ] + [1])
    
    print(ans)

if __name__ == "__main__":
    solve()