import sys
from functools import reduce

def solve():
    # Read input
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    H = list(map(int, input_data[1:]))

    # The condition "no building taller than Building j between i and j"
    # means Building j is a "visible" building from Building i looking right.
    # A building j is visible if H[j] > max(H[i+1...j-1]).
    # This is equivalent to saying Building j is part of the 
    # "upper envelope" of the heights to the right of i.
    
    # We can solve this by processing buildings from right to left.
    # We maintain a stack of indices of buildings that could be visible.
    # For a building i, the visible buildings j > i are:
    # 1. Building i+1
    # 2. The next building in the stack that is taller than the previous visible building.
    
    # However, a simpler observation:
    # Building j is visible from i if H[j] is a prefix maximum of the sequence H[i+1...N].
    # Let f(i) be the number of prefix maximums of H[i+1...N].
    # If H[i+1] is the tallest building in H[i+1...N], then f(i) = 1.
    # Otherwise, Building i+1 is the first prefix maximum, and the remaining
    # prefix maximums are those of the sequence H[i+2...N] that are taller than H[i+1].
    
    # Let next_taller[i] be the index of the first building j > i such that H[j] > H[i].
    # The number of visible buildings from i is:
    # c_i = 1 (for building i+1) + (c_{i+1} - number of buildings in H[i+2...N] 
    #                                   that are prefix maximums but shorter than H[i+1])
    # This is getting complex. Let's use the property:
    # c_i = 1 + c_{next_taller[i+1]} if H[i+1] is not the tallest.
    # Actually: c_i = 1 + (c_{next_taller[i+1]} if next_taller[i+1] exists else 0)
    # Wait, the condition is: j is visible from i if max(H[i+1...j-1]) < H[j].
    # This means H[j] must be a prefix maximum of the array starting at i+1.
    # Let g(i) be the number of prefix maximums of H[i...N].
    # g(i) = 1 + g(next_taller[i])
    # Then c_i = g(i+1).
    
    # 1. Find next_taller for all i using a stack
    # We process indices in reverse to find the first j > i with H[j] > H[i].
    def find_next_taller(indices, heights):
        stack = []
        next_taller = [N] * N
        for i in range(N - 1, -1, -1):
            while stack and heights[stack[-1]] < heights[i]:
                stack.pop()
            if stack:
                next_taller[i] = stack[-1]
            stack.append(i)
        return next_taller

    nt = find_next_taller(range(N), H)
    
    # 2. Calculate g(i) = 1 + g(nt[i]) using reduce from right to left
    # We use a list to store g values and update it. 
    # Since we can't use loops, we can use a dictionary or a list with reduce.
    # But we need the values of g for indices > i.
    # We can build the g array by reducing over the range(N-1, -1, -1).
    
    def calc_g(acc, i):
        # acc is a list where acc[j] stores g(j)
        # We need to compute g(i) = 1 + (acc[nt[i]] if nt[i] < N else 0)
        val = 1 + (acc[nt[i]] if nt[i] < N else 0)
        acc[i] = val
        return acc

    # Initialize acc list with 0s
    g_values = reduce(calc_g, range(N - 1, -1, -1), [0] * N)
    
    # c_i = g(i+1). For i=N, c_N = 0.
    # We shift g_values to the left by 1.
    result = g_values[1:] + [0]
    
    print(*(result))

if __name__ == "__main__":
    solve()