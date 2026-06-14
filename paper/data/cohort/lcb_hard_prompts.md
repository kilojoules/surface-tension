# LCB-hard audit pool — 108 problems, 40 contest weeks

Grouped by contest day. The audit picks at most one irreducible per contest week (cap rule from the brief — LCB-hard clusters graph/DP by contest, so don't let the off-diagonal cell become ten variants of the same algorithm).

## 2024-06-01  —  1 problem(s)

### `lcb/abc356_e`

```
You are given a sequence A=(A_1,\ldots,A_N) of length N.
Find \displaystyle \sum_{i=1}^{N-1}\sum_{j=i+1}^{N}\left\lfloor\frac{\max(A_i,A_j)}{\min(A_i,A_j)}\right\rfloor.
Here, \lfloor x \rfloor represents the greatest integer not greater than x. For example, \lfloor 3.14 \rfloor=3 and \lfloor 2 \rfloor=2.

Input

The input is given from Standard Input in the following format:
N
A_1 \ldots A_N

Output

Print the answer.

Constraints


- 2 \leq N \leq 2\times 10^5
- 1 \leq A_i \leq 10^6
- All input values are integers.

Sample Input 1

3
3 1 4

Sample Output 1

8

The sought value is
\left\lfloor\frac{\max(3,1)}{\min(3,1)}\right\rfloor + \left\lfloor\frac{\max(3,4)}{\min(3,4)}\right\rfloor + \left\lfloor\frac{\max(1,4)}{\min(1,4)}\right\rfloor\\ =\left\lfloor\frac{3}{1}\right\rfloor + \left\lfloor\frac{4}{3}\right\rfloor + \left\lfloor\frac{4}{1}\right\rfloor\\ =3+1+4\\ =8.

Sample Input 2

```

## 2024-06-08  —  1 problem(s)

### `lcb/abc357_e`

```
There is a directed graph with N vertices numbered 1 to N and N edges.
The out-degree of every vertex is 1, and the edge from vertex i points to vertex a_i.
Count the number of pairs of vertices (u, v) such that vertex v is reachable from vertex u.
Here, vertex v is reachable from vertex u if there exists a sequence of vertices w_0, w_1, \dots, w_K of length K+1 that satisfies the following conditions. In particular, if u = v, it is always reachable.

- w_0 = u.
- w_K = v.
- For every 0 \leq i \lt K, there is an edge from vertex w_i to vertex w_{i+1}.

Input

The input is given from Standard Input in the following format:
N
a_1 a_2 \dots a_N

Output

Print the number of pairs of vertices (u, v) such that vertex v is reachable from vertex u.

Constraints


- 1 \leq N \leq 2 \times 10^5
- 1 \leq a_i \leq N
- All input values are integers.

Sample Input 1

4
2 1 1 4

Sample Output 1

8

```

## 2024-06-15  —  1 problem(s)

### `lcb/abc358_e`

```
AtCoder Land sells tiles with English letters written on them. Takahashi is thinking of making a nameplate by arranging these tiles in a row.

Find the number, modulo 998244353, of strings consisting of uppercase English letters with a length between 1 and K, inclusive, that satisfy the following conditions:

- For every integer i satisfying 1 \leq i \leq 26, the following holds:
- Let a_i be the i-th uppercase English letter in lexicographical order. For example, a_1 =  A, a_5 =  E, a_{26} =  Z.
- The number of occurrences of a_i in the string is between 0 and C_i, inclusive.

Input

The input is given from Standard Input in the following format:
K
C_1 C_2 \ldots C_{26}

Output

Print the answer.

Constraints


- 1 \leq K \leq 1000
- 0 \leq C_i \leq 1000
- All input values are integers.

Sample Input 1

2
2 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0

Sample Output 1

10

The 10 strings that satisfy the conditions are A, B, C, AA, AB, AC, BA, BC, CA, CB.
```

## 2024-06-22  —  2 problem(s)

### `lcb/abc359_e`

```
You are given a sequence of positive integers of length N: H=(H _ 1,H _ 2,\dotsc,H _ N).
There is a sequence of non-negative integers of length N+1: A=(A _ 0,A _ 1,\dotsc,A _ N). Initially, A _ 0=A _ 1=\dotsb=A _ N=0.
Perform the following operations repeatedly on A:

- Increase the value of A _ 0 by 1.
- For i=1,2,\ldots,N in this order, perform the following operation:
- If A _ {i-1}\gt A _ i and A _ {i-1}\gt H _ i, decrease the value of A _ {i-1} by 1 and increase the value of A _ i by 1.



For each i=1,2,\ldots,N, find the number of operations before A _ i>0 holds for the first time.

Input

The input is given from Standard Input in the following format:
N
H _ 1 H _ 2 \dotsc H _ N

Output

Print the answers for i=1,2,\ldots,N in a single line, separated by spaces.

Constraints


- 1\leq N\leq2\times10 ^ 5
- 1\leq H _ i\leq10 ^ 9\ (1\leq i\leq N)
- All input values are integers.

Sample Input 1

5
3 1 4 1 5

Sample Output 1
```

### `lcb/abc359_d`

```
You are given a string S of length N consisting of characters A, B, and ?.
You are also given a positive integer K.
A string T consisting of A and B is considered a good string if it satisfies the following condition:

- No contiguous substring of length K in T is a palindrome.

Let q be the number of ? characters in S.
There are 2^q strings that can be obtained by replacing each ? in S with either A or B. Find how many of these strings are good strings.
The count can be very large, so find it modulo 998244353.

Input

The input is given from Standard Input in the following format:
N K
S

Output

Print the answer.

Constraints


- 2 \leq K \leq N \leq 1000
- K \leq 10
- S is a string consisting of A, B, and ?.
- The length of S is N.
- N and K are integers.

Sample Input 1

7 4
AB?A?BA

Sample Output 1
```

## 2024-06-30  —  1 problem(s)

### `lcb/abc360_e`

```
There are N - 1 white balls and one black ball. These N balls are arranged in a row, with the black ball initially at the leftmost position.
Takahashi will perform the following operation exactly K times.

- Choose an integer uniformly at random between 1 and N, inclusive, twice. Let a and b the chosen integers. If a \neq b, swap the a-th and b-th balls from the left.

After K operations, let the black ball be at the x-th position from the left. Find the expected value of x, modulo 998244353.


What is expected value modulo 998244353?

It can be proved that the sought expected value will always be rational. Additionally, under the constraints of this problem, it can be proved that if this value is expressed as an irreducible fraction \frac{P}{Q}, then Q \not \equiv 0 \pmod{998244353}. Therefore, there exists a unique integer R such that R \times Q \equiv P \pmod{998244353}, 0 \leq R < 998244353. Report this R.

Input

The input is given from Standard Input in the following format:
N K

Output

Print the answer in one line.

Constraints


- 1 \leq N \leq 998244352
- 1 \leq K \leq 10^5

Sample Input 1

2 1

Sample Output 1

499122178

```

## 2024-07-06  —  3 problem(s)

### `lcb/abc361_d`

```
There are N+2 cells arranged in a row. Let cell i denote the i-th cell from the left.
There is one stone placed in each of the cells from cell 1 to cell N.
For each 1 \leq i \leq N, the stone in cell i is white if S_i is W, and black if S_i is B.
Cells N+1 and N+2 are empty.
You can perform the following operation any number of times (possibly zero):

- Choose a pair of adjacent cells that both contain stones, and move these two stones to the empty two cells while preserving their order.
  More precisely, choose an integer x such that 1 \leq x \leq N+1 and both cells x and x+1 contain stones. Let k and k+1 be the empty two cells. Move the stones from cells x and x+1 to cells k and k+1, respectively.

Determine if it is possible to achieve the following state, and if so, find the minimum number of operations required:

- Each of the cells from cell 1 to cell N contains one stone, and for each 1 \leq i \leq N, the stone in cell i is white if T_i is W, and black if T_i is B.

Input

The input is given from Standard Input in the following format:
N
S
T

Output

If it is possible to achieve the desired state, print the minimum number of operations required. If it is impossible, print -1.

Constraints


- 2 \leq N \leq 14
- N is an integer.
- Each of S and T is a string of length N consisting of B and W.

Sample Input 1

6
BWBWBW
```

### `lcb/abc361_f`

```
How many integers x between 1 and N, inclusive, can be expressed as x = a^b using some positive integer a and a positive integer b not less than 2?

Input

The input is given from Standard Input in the following format:
N

Output

Print the answer as an integer.

Constraints


- All input values are integers.
- 1 \le N \le 10^{18}

Sample Input 1

99

Sample Output 1

12

The integers that satisfy the conditions in the problem statement are 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, 64, 81: there are 12.

Sample Input 2

1000000000000000000

Sample Output 2

1001003332

```

### `lcb/abc361_e`

```
In the nation of AtCoder, there are N cities numbered 1 to N and N-1 roads numbered 1 to N-1.
Road i connects cities A_i and B_i bidirectionally, and its length is C_i. Any pair of cities can be reached from each other by traveling through some roads.
Find the minimum travel distance required to start from a city and visit all cities at least once using the roads.

Input

The input is given from Standard Input in the following format:
N
A_1 B_1 C_1
\vdots
A_{N-1} B_{N-1} C_{N-1}

Output

Print the answer.

Constraints


- 2 \leq N \leq 2\times 10^5
- 1 \leq A_i, B_i \leq N
- 1 \leq C_i \leq 10^9
- All input values are integers.
- Any pair of cities can be reached from each other by traveling through some roads.

Sample Input 1

4
1 2 2
1 3 3
1 4 4

Sample Output 1

11
```

## 2024-07-13  —  2 problem(s)

### `lcb/abc362_d`

```
You are given a simple connected undirected graph with N vertices and M edges. Each vertex i\,(1\leq i \leq N) has a weight A_i. Each edge j\,(1\leq j \leq M) connects vertices U_j and V_j bidirectionally and has a weight B_j.
The weight of a path in this graph is defined as the sum of the weights of the vertices and edges that appear on the path.
For each i=2,3,\dots,N, solve the following problem:

- Find the minimum weight of a path from vertex 1 to vertex i.

Input

The input is given from Standard Input in the following format:
N M
A_1 A_2 \dots A_N
U_1 V_1 B_1
U_2 V_2 B_2
\vdots
U_M V_M B_M

Output

Print the answers for i=2,3,\dots,N in a single line, separated by spaces.

Constraints


- 2 \leq N \leq 2 \times 10^5
- N-1 \leq M \leq 2 \times 10^5
- 1 \leq U_j < V_j \leq N
- (U_i, V_i) \neq (U_j, V_j) if i \neq j.
- The graph is connected.
- 0 \leq A_i \leq 10^9
- 0 \leq B_j \leq 10^9
- All input values are integers.

Sample Input 1

3 3
```

### `lcb/abc362_e`

```
You are given a sequence A = (A_1, A_2, \dots, A_N) of length N. For each k = 1, 2, \dots, N, find the number, modulo 998244353, of (not necessarily contiguous) subsequences of A of length k that are arithmetic sequences. Two subsequences are distinguished if they are taken from different positions, even if they are equal as sequences.

What is a subsequence?
A subsequence of a sequence A is a sequence obtained by deleting zero or more elements from A and arranging the remaining elements without changing the order.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \dots A_N

Output

Print the answers for k = 1, 2, \dots, N in this order, in a single line, separated by spaces.

Constraints


- 1 \leq N \leq 80
- 1 \leq A_i \leq 10^9
- All input values are integers.

Sample Input 1

5
1 2 3 2 3

Sample Output 1

5 10 3 0 0


- There are 5 subsequences of length 1, all of which are arithmetic sequences.
- There are 10 subsequences of length 2, all of which are arithmetic sequences.
- There are 3 subsequences of length 3 that are arithmetic sequences: (A_1, A_2, A_3), (A_1, A_2, A_5), and (A_1, A_4, A_5).
```

## 2024-07-20  —  2 problem(s)

### `lcb/abc363_f`

```
You are given an integer N. Print a string S that satisfies all of the following conditions. If no such string exists, print -1.

- S is a string of length between 1 and 1000, inclusive, consisting of the characters 1, 2, 3, 4, 5, 6, 7, 8, 9, and * (multiplication symbol).
- S is a palindrome.
- The first character of S is a digit.
- The value of S when evaluated as a formula equals N.

Input

The input is given from Standard Input in the following format:
N

Output

If there is a string S that satisfies the conditions exists, print such a string. Otherwise, print -1.

Constraints


- 1 \leq N \leq 10^{12}
- N is an integer.

Sample Input 1

363

Sample Output 1

11*3*11

S = 11*3*11 satisfies the conditions in the problem statement. Another string that satisfies the conditions is S= 363.

Sample Input 2

101
```

### `lcb/abc363_e`

```
There is an island of size H \times W, surrounded by the sea.
The island is divided into H rows and W columns of 1 \times 1 sections, and the elevation of the section at the i-th row from the top and the j-th column from the left (relative to the current sea level) is A_{i,j}.
Starting from now, the sea level rises by 1 each year.
Here, a section that is vertically or horizontally adjacent to the sea or a section sunk into the sea and has an elevation not greater than the sea level will sink into the sea.
Here, when a section newly sinks into the sea, any vertically or horizontally adjacent section with an elevation not greater than the sea level will also sink into the sea simultaneously, and this process repeats for the newly sunk sections.
For each i=1,2,\ldots, Y, find the area of the island that remains above sea level i years from now.

Input

The input is given from Standard Input in the following format:
H W Y
A_{1,1} A_{1,2} \ldots A_{1,W}
A_{2,1} A_{2,2} \ldots A_{2,W}
\vdots
A_{H,1} A_{H,2} \ldots A_{H,W}

Output

Print Y lines.
The i-th line (1 \leq i \leq Y) should contain the area of the island that remains above sea level i years from now.

Constraints


- 1 \leq H, W \leq 1000
- 1 \leq Y \leq 10^5
- 1 \leq A_{i,j} \leq 10^5
- All input values are integers.

Sample Input 1

3 3 5
10 2 10
3 1 4
10 5 10
```

## 2024-07-27  —  3 problem(s)

### `lcb/abc364_e`

```
Takahashi has prepared N dishes for Snuke.
The dishes are numbered from 1 to N, and dish i has a sweetness of A_i and a saltiness of B_i.
Takahashi can arrange these dishes in any order he likes.
Snuke will eat the dishes in the order they are arranged, but if at any point the total sweetness of the dishes he has eaten so far exceeds X or the total saltiness exceeds Y, he will not eat any further dishes.
Takahashi wants Snuke to eat as many dishes as possible.
Find the maximum number of dishes Snuke will eat if Takahashi arranges the dishes optimally.

Input

The input is given from Standard Input in the following format:
N X Y
A_1 B_1
A_2 B_2
\vdots
A_N B_N

Output

Print the answer as an integer.

Constraints


- 1 \leq N \leq 80
- 1 \leq A_i, B_i \leq 10000
- 1 \leq X, Y \leq 10000
- All input values are integers.

Sample Input 1

4 8 4
1 5
3 2
4 1
5 3
```

### `lcb/abc364_f`

```
There is a graph with N + Q vertices, numbered 1, 2, \ldots, N + Q. Initially, the graph has no edges.
For this graph, perform the following operation for i = 1, 2, \ldots, Q in order:

- For each integer j satisfying L_i \leq j \leq R_i, add an undirected edge with cost C_i between vertices N + i and j.

Determine if the graph is connected after all operations are completed. If it is connected, find the cost of a minimum spanning tree of the graph.
A minimum spanning tree is a spanning tree with the smallest possible cost, and the cost of a spanning tree is the sum of the costs of the edges used in the spanning tree.

Input

The input is given from Standard Input in the following format:
N Q
L_1 R_1 C_1
L_2 R_2 C_2
\vdots
L_Q R_Q C_Q

Output

If the graph is connected, print the cost of a minimum spanning tree. Otherwise, print -1.

Constraints


- 1 \leq N, Q \leq 2 \times 10^5
- 1 \leq L_i \leq R_i \leq N
- 1 \leq C_i \leq 10^9
- All input values are integers.

Sample Input 1

4 3
1 2 2
1 3 4
2 4 5
```

### `lcb/abc364_d`

```
There are N+Q points A_1,\dots,A_N,B_1,\dots,B_Q on a number line, where point A_i has a coordinate a_i and point B_j has a coordinate b_j.
For each j=1,2,\dots,Q, answer the following question:

- Let X be the point among A_1,A_2,\dots,A_N that is the k_j-th closest to point B_j. Find the distance between points X and B_j.
More formally, let d_i be the distance between points A_i and B_j. Sort (d_1,d_2,\dots,d_N) in ascending order to get the sequence (d_1',d_2',\dots,d_N'). Find d_{k_j}'.

Input

The input is given from Standard Input in the following format:
N Q
a_1 a_2 \dots a_N
b_1 k_1
b_2 k_2
\vdots
b_Q k_Q

Output

Print Q lines.
The l-th line (1 \leq l \leq Q) should contain the answer to the question for j=l as an integer.

Constraints


- 1 \leq N, Q \leq 10^5
- -10^8 \leq a_i, b_j \leq 10^8
- 1 \leq k_j \leq N
- All input values are integers.

Sample Input 1

4 3
-3 -1 5 6
-2 3
2 1
```

## 2024-08-03  —  1 problem(s)

### `lcb/abc365_e`

```
You are given an integer sequence A=(A_1,\ldots,A_N) of length N. Find the value of the following expression:
\displaystyle \sum_{i=1}^{N-1}\sum_{j=i+1}^N (A_i \oplus A_{i+1}\oplus \ldots \oplus A_j).

Notes on bitwise XOR
The bitwise XOR of non-negative integers A and B, denoted as A \oplus B, is defined as follows:
- In the binary representation of A \oplus B, the digit at the 2^k (k \geq 0) position is 1 if and only if exactly one of the digits at the 2^k position in the binary representations of A and B is 1; otherwise, it is 0.
For example, 3 \oplus 5 = 6 (in binary: 011 \oplus 101 = 110).
In general, the bitwise XOR of k integers p_1, \dots, p_k is defined as (\cdots ((p_1 \oplus p_2) \oplus p_3) \oplus \cdots \oplus p_k).  It can be proved that this is independent of the order of p_1, \dots, p_k.

Input

The input is given from Standard Input in the following format:
N 
A_1 A_2 \ldots A_{N}

Output

Print the answer.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 1 \leq A_i \leq 10^8
- All input values are integers.

Sample Input 1

3
1 3 2

Sample Output 1

3

```

## 2024-08-04  —  3 problem(s)

### `lcb/arc181_d`

```
You are given a permutation P=(P_1,P_2,\dots,P_N) of (1,2,\dots,N).
Consider the following operations k\ (k=2,3,\dots,N) on this permutation.

- Operation k: For i=1,2,\dots,k-1 in this order, if P_i > P_{i+1}, swap the values of the i-th and (i+1)-th elements of P.

You are also given a non-decreasing sequence A=(A_1,A_2,\dots,A_M)\ (2 \leq A_i \leq N) of length M.
For each i=1,2,\dots,M, find the inversion number of P after applying the operations A_1, A_2, \dots, A_i in this order.

 What is the inversion number of a sequence?

The inversion number of a sequence x=(x_1,x_2,\dots,x_n) of length n is the number of pairs of integers (i,j)\ (1\leq i < j \leq n) such that x_i > x_j.

Input

The input is given from Standard Input in the following format:
N
P_1 P_2 \dots P_N
M
A_1 A_2 \dots A_M

Output

Print M lines. The k-th line should contain the answer to the problem for i=k.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 1 \leq M \leq 2 \times 10^5
- 2 \leq A_i \leq N
- P is a permutation of (1,2,\dots,N).
- A_i \leq A_{i+1} for i=1,2,\dots,M-1.
- All input values are integers.

Sample Input 1
```

### `lcb/arc181_c`

```
You are given two permutations P=(P_1,P_2,\dots,P_N) and Q=(Q_1,Q_2,\dots,Q_N) of (1,2,\dots,N).
Write one of the characters 0 and 1 in each cell of an N-by-N grid so that all of the following conditions are satisfied:

- Let S_i be the string obtained by concatenating the characters in the i-th row from the 1-st to the N-th column. Then, S_{P_1} < S_{P_2} < \dots < S_{P_N} in lexicographical order.
- Let T_i be the string obtained by concatenating the characters in the i-th column from the 1-st to the N-th row. Then, T_{Q_1} < T_{Q_2} < \dots < T_{Q_N} in lexicographical order.

It can be proved that for any P and Q, there is at least one way to write the characters that satisfies all the conditions.
 What does "X < Y in lexicographical order" mean?
For strings X=X_1X_2\dots X_{|X|} and Y = Y_1Y_2\dots Y_{|Y|}, "X < Y in lexicographical order" means that 1. or 2. below holds.
Here, |X| and |Y| denote the lengths of X and Y, respectively.

-  |X| \lt |Y| and X_1X_2\ldots X_{|X|} = Y_1Y_2\ldots Y_{|X|}. 
-  There exists an integer 1 \leq i \leq \min\lbrace |X|, |Y| \rbrace such that both of the following are true:

-  X_1X_2\ldots X_{i-1} = Y_1Y_2\ldots Y_{i-1}
-  X_i is less than Y_i.

Input

The input is given from Standard Input in the following format:
N
P_1 P_2 \dots P_N
Q_1 Q_2 \dots Q_N

Output

Print a way to fill the grid that satisfies the conditions in the following format, where A_{ij} is the character written at the i-th row and j-th column:
A_{11}A_{12}\dots A_{1N}
\vdots
A_{N1}A_{N2}\dots A_{NN}

If there are multiple ways to satisfy the conditions, any of them will be accepted.

Constraints

```

### `lcb/arc181_b`

```
For strings S and T consisting of lowercase English letters, and a string X consisting of 0 and 1, define the string f(S,T,X) consisting of lowercase English letters as follows:

- Starting with an empty string, for each i=1,2,\dots,|X|, append S to the end if the i-th character of X is 0, and append T to the end if it is 1.

You are given a string S consisting of lowercase English letters, and strings X and Y consisting of 0 and 1.
Determine if there exists a string T (which can be empty) such that f(S,T,X)=f(S,T,Y).
You have t test cases to solve.

Input

The input is given from Standard Input in the following format:
t
\mathrm{case}_1
\vdots
\mathrm{case}_t

Each case is given in the following format:
S
X
Y

Output

Print t lines. The i-th line should contain Yes if there exists a T that satisfies the condition for the i-th test case, and No otherwise.

Constraints


- 1 \leq t \leq 5 \times 10^5
- 1 \leq |S| \leq 5\times 10^5
- 1 \leq |X|,|Y| \leq 5\times 10^5
- S is a string consisting of lowercase English letters.
- X and Y are strings consisting of 0 and 1.
- The sum of |S| across all test cases in a single input is at most 5 \times 10^5.
- The sum of |X| across all test cases in a single input is at most 5 \times 10^5.
```

## 2024-08-10  —  3 problem(s)

### `lcb/abc366_f`

```
You are given N linear functions f_1, f_2, \ldots, f_N, where f_i(x) = A_i x + B_i.
Find the maximum possible value of f_{p_1}(f_{p_2}(\ldots f_{p_K}(1) \ldots )) for a sequence p = (p_1, p_2, \ldots, p_K) of K distinct integers between 1 and N, inclusive.

Input

The input is given from Standard Input in the following format:
N K
A_1 B_1
A_2 B_2
\vdots
A_N B_N

Output

Print the answer as an integer.

Constraints


- 1 \leq N \leq 2 \times 10^{5}
- 1 \leq K \leq \text{min}(N,10)
- 1 \leq A_i, B_i \leq 50 (1 \leq i \leq N)
- All input values are integers.

Sample Input 1

3 2
2 3
1 5
4 2

Sample Output 1

26

```

### `lcb/abc366_e`

```
You are given N points (x_1, y_1), (x_2, y_2), \dots, (x_N, y_N) on a two-dimensional plane, and a non-negative integer D.
Find the number of integer pairs (x, y) such that \displaystyle \sum_{i=1}^N (|x-x_i|+|y-y_i|) \leq D.

Input

The input is given from Standard Input in the following format:
N D
x_1 y_1
x_2 y_2
\vdots
x_N y_N

Output

Print the answer.

Constraints


- 1 \leq N \leq 2 \times 10^5
- 0 \leq D \leq 10^6
- -10^6 \leq x_i, y_i \leq 10^6
- (x_i, y_i) \neq (x_j, y_j) for i \neq j.
- All input values are integers.

Sample Input 1

2 3
0 0
1 0

Sample Output 1

8

```

### `lcb/abc366_g`

```
You are given a simple undirected graph with N vertices and M edges. The i-th edge connects vertices u_i and v_i bidirectionally.
Determine if there exists a way to write an integer between 1 and 2^{60} - 1, inclusive, on each vertex of this graph so that the following condition is satisfied:

- For every vertex v with a degree of at least 1, the total XOR of the numbers written on its adjacent vertices (excluding v itself) is 0.


What is XOR?

The XOR of two non-negative integers A and B, denoted as A \oplus B, is defined as follows:


- In the binary representation of A \oplus B, the bit at position 2^k \, (k \geq 0) is 1 if and only if exactly one of the bits at position 2^k in the binary representations of A and B is 1. Otherwise, it is 0.


For example, 3 \oplus 5 = 6 (in binary: 011 \oplus 101 = 110).

In general, the bitwise XOR of k integers p_1, \dots, p_k is defined as (\cdots ((p_1 \oplus p_2) \oplus p_3) \oplus \cdots \oplus p_k).  It can be proved that this is independent of the order of p_1, \dots, p_k.

Input

The input is given from Standard Input in the following format:
N M
u_1 v_1
u_2 v_2
\vdots
u_M v_M

Output

If there is no way to write integers satisfying the condition, print No.
Otherwise, let X_v be the integer written on vertex v, and print your solution in the following format. If multiple solutions exist, any of them will be accepted.
Yes
X_1 X_2 \dots X_N

Constraints
```

## 2024-08-11  —  4 problem(s)

### `lcb/arc182_d`

```
An integer sequence where no two adjacent elements are the same is called a good sequence.
You are given two good sequences of length N: A=(A_1,A_2,\dots,A_N) and B=(B_1,B_2,\dots,B_N). Each element of A and B is between 0 and M-1, inclusive.
You can perform the following operations on A any number of times, possibly zero:

- Choose an integer i between 1  and N, inclusive, and perform one of the following:
- Set A_i \leftarrow (A_i + 1) \bmod M.
- Set A_i \leftarrow (A_i - 1) \bmod M. Here, (-1) \bmod M = M - 1.



However, you cannot perform an operation that makes A no longer a good sequence.
Determine if it is possible to make A equal to B, and if it is possible, find the minimum number of operations required to do so.

Input

The input is given from Standard Input in the following format:
N M
A_1 A_2 \dots A_N
B_1 B_2 \dots B_N

Output

If the goal is unachievable, print -1.
Otherwise, print the minimum number of operations required as an integer.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 2 \leq M \leq 10^6
- 0\leq A_i,B_i< M(1\leq i\leq N)
- A_i\ne A_{i+1}(1\leq i\leq N-1)
- B_i\ne B_{i+1}(1\leq i\leq N-1)
- All input values are integers.

```

### `lcb/arc182_e`

```
You are given positive integers N, M, K, a non-negative integer C, and an integer sequence A=(A_1, A_2, \ldots, A_N) of length N.
Find \displaystyle \sum_{k=0}^{K-1}\min_{1\le i\le N}\lbrace(Ck+A_i)\ \mathrm{mod}\ M \rbrace.

Input

The input is given from Standard Input in the following format:
N M C K
A_1 A_2 \ldots A_N

Output

Print the answer.

Constraints


- 1 \le N \le 10^5
- 1 \le M \le 10^9
- 0 \le C < M
- 1 \le K \le 10^9
- 0 \le A_i < M
- All input values are integers.

Sample Input 1

2 5 3 3
1 3

Sample Output 1

4

For k=0, \lbrace(3k+1)\ \mathrm{mod}\ 5 \rbrace=1 and \lbrace(3k+3)\ \mathrm{mod}\ 5 \rbrace=3, so \displaystyle \min_{1\le i\le N}\lbrace(Ck+A_i)\ \mathrm{mod}\ M \rbrace=1.
For k=1, \lbrace(3k+1)\ \mathrm{mod}\ 5 \rbrace=4 and \lbrace(3k+3)\ \mathrm{mod}\ 5 \rbrace=1, so \displaystyle \min_{1\le i\le N}\lbrace(Ck+A_i)\ \mathrm{mod}\ M \rbrace=1.
For k=2, \lbrace(3k+1)\ \mathrm{mod}\ 5 \rbrace=2 and \lbrace(3k+3)\ \mathrm{mod}\ 5 \rbrace=4, so \displaystyle \min_{1\le i\le N}\lbrace(Ck+A_i)\ \mathrm{mod}\ M \rbrace=2.
```

### `lcb/arc182_a`

```
There is an integer sequence S of length N. Initially, all elements of S are 0.
You are also given two integer sequences of length Q: P=(P_1,P_2,\dots,P_Q) and V=(V_1,V_2,\dots,V_Q).
Snuke wants to perform Q operations on the sequence S in order. The i-th operation is as follows:

- Perform one of the following:
- Replace each of the elements S_1, S_2, \dots, S_{P_i} with V_i. However, before this operation, if there is an element among S_1, S_2, \dots, S_{P_i} that is strictly greater than V_i, Snuke will start crying.
- Replace each of the elements S_{P_i}, S_{P_i+1}, \dots, S_N with V_i. However, before this operation, if there is an element among S_{P_i}, S_{P_i+1}, \dots, S_N that is strictly greater than V_i, Snuke will start crying.



Find the number of sequences of Q operations where Snuke can perform all operations without crying, modulo 998244353.
Two sequences of operations are distinguished if and only if there is 1 \leq i \leq Q such that the choice for the i-th operation is different.

Input

The input is given from Standard Input in the following format:
N Q
P_1 V_1
P_2 V_2
\vdots
P_Q V_Q

Output

Print the answer as an integer.

Constraints


- 2 \leq N \leq 5000
- 1 \leq Q \leq 5000
- 1 \leq P_i \leq N
- 1 \leq V_i \leq 10^9
- All input values are integers.

```

### `lcb/arc182_c`

```
An integer sequence of length between 1 and N, inclusive, where each element is between 1 and M, inclusive, is called a good sequence.
The score of a good sequence is defined as the number of positive divisors of X, where X is the product of the elements in the sequence.
There are \displaystyle \sum_{k=1}^{N}M^k good sequences. Find the sum of the scores of all those sequences modulo 998244353.

Input

The input is given from Standard Input in the following format:
N M

Output

Print the answer as an integer.

Constraints


- 1 \leq N \leq 10^{18}
- 1 \leq M \leq 16
- All input values are integers.

Sample Input 1

1 7

Sample Output 1

16

There are seven good sequences: (1),(2),(3),(4),(5),(6),(7). Their scores are 1,2,2,3,2,4,2, respectively, so the answer is 1+2+2+3+2+4+2=16.

Sample Input 2

3 11

Sample Output 2
```

## 2024-08-17  —  3 problem(s)

### `lcb/abc367_e`

```
You are given a sequence X of length N where each element is between 1 and N, inclusive, and a sequence A of length N.
Print the result of performing the following operation K times on A.

- Replace A with B such that B_i = A_{X_i}.

Input

The input is given from Standard Input in the following format:
N K
X_1 X_2 \dots X_N
A_1 A_2 \dots A_N

Output

Let A' be the sequence A after the operations. Print it in the following format:
A'_1 A'_2 \dots A'_N

Constraints


- All input values are integers.
- 1 \le N \le 2 \times 10^5
- 0 \le K \le 10^{18}
- 1 \le X_i \le N
- 1 \le A_i \le 2 \times 10^5

Sample Input 1

7 3
5 2 6 3 1 4 6
1 2 3 5 7 9 11

Sample Output 1

7 2 3 5 1 9 3
```

### `lcb/abc367_f`

```
You are given sequences of positive integers of length N: A=(A_1,A_2,\ldots,A_N) and B=(B_1,B_2,\ldots,B_N).
You are given Q queries to process in order. The i-th query is explained below.

- You are given positive integers l_i,r_i,L_i,R_i. Print Yes if it is possible to rearrange the subsequence (A_{l_i},A_{l_i+1},\ldots,A_{r_i}) to match the subsequence (B_{L_i},B_{L_i+1},\ldots,B_{R_i}), and No otherwise.

Input

The input is given from Standard Input in the following format:
N Q
A_1 A_2 \ldots A_N
B_1 B_2 \ldots B_N
l_1 r_1 L_1 R_1
l_2 r_2 L_2 R_2
\vdots
l_Q r_Q L_Q R_Q

Output

Print Q lines. The i-th line should contain the answer to the i-th query.

Constraints


-  1\leq N,Q\leq 2\times 10^5
-  1\leq A_i,B_i\leq N
-  1\leq l_i \leq r_i\leq N
-  1\leq L_i \leq R_i\leq N
- All input values are integers.

Sample Input 1

5 4
1 2 3 2 4
2 3 1 4 2
1 3 1 3
```

### `lcb/abc367_g`

```
You are given positive integers N, M, K, and a sequence of non-negative integers: A=(A_1,A_2,\ldots,A_N).
For a non-empty non-negative integer sequence B=(B_1,B_2,\ldots,B_{|B|}), we define its score as follows.

- If the length of B is a multiple of M: (B_1 \oplus B_2 \oplus \dots \oplus B_{|B|})^K
- Otherwise: 0

Here, \oplus represents the bitwise XOR.
Find the sum, modulo 998244353, of the scores of the 2^N-1 non-empty subsequences of A.
What is bitwise XOR? The bitwise XOR of non-negative integers A and B, denoted as A \oplus B, is defined as follows: - In the binary representation of A \oplus B, the digit at position 2^k (k \geq 0) is 1 if exactly one of A and B has a 1 in that position in their binary representations, and 0 otherwise. For example, 3 \oplus 5 = 6 (in binary: 011 \oplus 101 = 110). In general, the XOR of k integers p_1, \dots, p_k is defined as (\cdots ((p_1 \oplus p_2) \oplus p_3) \oplus \cdots \oplus p_k), and it can be proved that this is independent of the order of p_1, \dots, p_k.

Input

The input is given from Standard Input in the following format:
N M K
A_1 A_2 \ldots A_N

Output

Print the answer.

Constraints


- 1 \leq N,K \leq 2 \times 10^5
- 1 \leq M \leq 100
- 0 \leq A_i < 2^{20}
- All input values are integers.

Sample Input 1

3 2 2
1 2 3

Sample Output 1

```

## 2024-08-24  —  4 problem(s)

### `lcb/abc368_g`

```
You are given sequences of positive integers A and B of length N. Process Q queries given in the following forms in the order they are given. Each query is of one of the following three types.

- 
Type 1: Given in the form 1 i x. Replace A_i with x.

- 
Type 2: Given in the form 2 i x. Replace B_i with x.

- 
Type 3: Given in the form 3 l r. Solve the following problem and print the answer.

- 
Initially, set v = 0. For i = l, l+1, ..., r in this order, replace v with either v + A_i or v \times B_i. Find the maximum possible value of v at the end.




It is guaranteed that the answers to the given type 3 queries are at most 10^{18}.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \cdots A_N
B_1 B_2 \cdots B_N
Q
query_1
query_2
\vdots
query_Q

Here, query_i is the i-th query, given in one of the following formats:
1 i x

2 i x
```

### `lcb/abc368_f`

```
You are given a sequence of N positive integers A = (A_1, A_2, \dots ,A_N), where each element is at least 2. Anna and Bruno play a game using these integers. They take turns, with Anna going first, performing the following operation.

- Choose an integer i \ (1 \leq i \leq N) freely. Then, freely choose a positive divisor x of A_i that is not A_i itself, and replace A_i with x.

The player who cannot perform the operation loses, and the other player wins. Determine who wins assuming both players play optimally for victory.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \cdots A_N

Output

Print Anna if Anna wins the game, and Bruno if Bruno wins.

Constraints


- 1 \leq N \leq 10^5
- 2 \leq A_i \leq 10^5
- All input values are integers.

Sample Input 1

3
2 3 4

Sample Output 1

Anna

For example, the game might proceed as follows. Note that this example may not necessarily represent optimal play by both players:

- Anna changes A_3 to 2.
```

### `lcb/abc368_d`

```
You are given a tree with N vertices numbered 1 to N. The i-th edge connects vertices A_i and B_i.
Consider a tree that can be obtained by removing some (possibly zero) edges and vertices from this graph. Find the minimum number of vertices in such a tree that includes all of K specified vertices V_1,\ldots,V_K.

Input

The input is given from Standard Input in the following format:
N K
A_1 B_1
\vdots
A_{N-1} B_{N-1}
V_1 \ldots V_K

Output

Print the answer.

Constraints


- 1 \leq K \leq N \leq 2\times 10^5
- 1 \leq A_i,B_i \leq N
- 1 \leq V_1 < V_2 < \ldots < V_K \leq N
- The given graph is a tree.
- All input values are integers.

Sample Input 1

7 3
1 2
1 3
2 4
2 5
3 6
3 7
1 3 5
```

### `lcb/abc368_e`

```
In the nation of Atcoder, there are N cities numbered 1 to N, and M trains numbered 1 to M.
Train i departs from city A_i at time S_i and arrives at city B_i at time T_i.
Given a positive integer X_1, find a way to set non-negative integers X_2,\ldots,X_M that satisfies the following condition with the minimum possible value of X_2+\ldots+X_M.

- Condition: For all pairs (i,j) satisfying 1 \leq i,j \leq M, if B_i=A_j and T_i \leq S_j, then T_i+X_i \leq S_j+X_j.
- In other words, for any pair of trains that are originally possible to transfer between, it is still possible to transfer even after delaying the departure and arrival times of each train i by X_i.



It can be proved that such a way to set X_2,\ldots,X_M with the minimum possible value of X_2+\ldots+X_M is unique.

Input

The input is given from Standard Input in the following format:
N M X_1
A_1 B_1 S_1 T_1
\vdots
A_M B_M S_M T_M

Output

Print X_2,\ldots,X_M that satisfy the condition with the minimum possible sum, in that order, separated by spaces.

Constraints


- 2 \leq N \leq 2\times 10^5
- 2 \leq M \leq 2\times 10^5
- 1 \leq A_i,B_i \leq N
- A_i \neq B_i
- 0 \leq S_i < T_i \leq 10^9
- 1 \leq X_1 \leq 10^9
- All input values are integers.

Sample Input 1
```

## 2024-08-25  —  3 problem(s)

### `lcb/arc183_b`

```
You are given integer sequences of length N: A=(A_1,A_2,\cdots,A_N) and B=(B_1,B_2,\cdots,B_N), and an integer K.
You can perform the following operation zero or more times.

- Choose integers i and j (1 \leq i,j \leq N).
Here, |i-j| \leq K must hold.
Then, change the value of A_i to A_j.

Determine whether it is possible to make A identical to B.
There are T test cases for each input.

Input

The input is given from Standard Input in the following format:
T
case_1
case_2
\vdots
case_T

Each test case is given in the following format:
N K
A_1 A_2 \cdots A_N
B_1 B_2 \cdots B_N

Output

For each test case, print Yes if it is possible to make A identical to B, and No otherwise.

Constraints


- 1 \leq T \leq 125000
- 1 \leq K < N \leq 250000
- 1 \leq A_i,B_i \leq N
- The sum of N across all test cases in each input is at most 250000.
```

### `lcb/arc183_c`

```
Find the number, modulo 998244353, of permutations P=(P_1,P_2,\cdots,P_N) of (1,2,\cdots,N) that satisfy all of the following M conditions.

- The i-th condition: The maximum among P_{L_i},P_{L_i+1},\cdots,P_{R_i} is not P_{X_i}.
Here, L_i, R_i, and X_i are integers given in the input.

Input

The input is given from Standard Input in the following format:
N M
L_1 R_1 X_1
L_2 R_2 X_2
\vdots
L_M R_M X_M

Output

Print the answer.

Constraints


- 1 \leq N \leq 500
- 1 \leq M \leq 10^5
- 1 \leq L_i \leq X_i \leq R_i \leq N
- All input values are integers.

Sample Input 1

3 2
1 3 2
1 2 1

Sample Output 1

1
```

### `lcb/arc183_d`

```
There is a tree with N vertices numbered from 1 to N.
The i-th edge connects vertices A_i and B_i.
Here, N is even, and furthermore, this tree has a perfect matching.
Specifically, for each i (1 \leq i \leq N/2), it is guaranteed that A_i=i \times 2-1 and B_i=i \times 2.
You will perform the following operation N/2 times:

- Choose two leaves (vertices with degree exactly 1) and remove them from the tree.
Here, the tree after removal must still have a perfect matching.
In this problem, we consider a graph with zero vertices to be a tree as well.

For each operation, its score is defined as the distance between the two chosen vertices (the number of edges on the simple path connecting the two vertices).
Show one procedure that maximizes the total score.
It can be proved that there always exists a procedure to complete N/2 operations under the constraints of this problem.

Input

The input is given from Standard Input in the following format:
N
A_1 B_1
A_2 B_2
\vdots
A_{N-1} B_{N-1}

Output

Print a solution in the following format:
X_1 Y_1
X_2 Y_2
\vdots
X_{N/2} Y_{N/2}

Here, X_i and Y_i are the two vertices chosen in the i-th operation.
If there are multiple solutions, you may print any of them.

Constraints
```

## 2024-08-31  —  2 problem(s)

### `lcb/abc369_g`

```
You are given a tree with N vertices.
The vertices are numbered 1, 2, \ldots, N.
The i-th edge (1\leq i\leq N-1) connects vertices U_i and V_i, with a length of L_i.
For each K=1,2,\ldots, N, solve the following problem.

Takahashi and Aoki play a game. The game proceeds as follows.

- First, Aoki specifies K distinct vertices on the tree.
- Then, Takahashi constructs a walk that starts and ends at vertex 1, and passes through all the vertices specified by Aoki.

The score is defined as the length of the walk constructed by Takahashi. Takahashi wants to minimize the score, while Aoki wants to maximize it.
Find the score when both players play optimally.


Definition of a walk
    A walk on an undirected graph (possibly a tree) is a sequence of k vertices and k-1 edges v_1,e_1,v_2,\ldots,v_{k-1},e_{k-1},v_k (where k is a positive integer)
    such that edge e_i connects vertices v_i and v_{i+1}. The same vertex or edge can appear multiple times in the sequence.  
    A walk is said to pass through vertex x if there exists at least one i (1\leq i\leq k) such that v_i=x. (There can be multiple such i.)  
    The walk is said to start and end at v_1 and v_k, respectively, and the length of the walk is the sum of the lengths of e_1, e_2, \ldots, e_{k-1}.

Input

The input is given from Standard Input in the following format:
N
U_1 V_1 L_1
U_2 V_2 L_2
\vdots
U_{N-1} V_{N-1} L_{N-1}

Output

Print N lines.
The i-th line (1\leq i\leq N) should contain the answer to the problem for K=i.

Constraints
```

### `lcb/abc369_e`

```
There are N islands and M bidirectional bridges connecting two islands. The islands and bridges are numbered 1, 2, \ldots, N and 1, 2, \ldots, M, respectively.
Bridge i connects islands U_i and V_i, and the time it takes to cross it in either direction is T_i.
No bridge connects an island to itself, but it is possible for two islands to be directly connected by more than one bridge.
One can travel between any two islands using some bridges.
You are given Q queries, so answer each of them. The i-th query is as follows:

You are given K_i distinct bridges: bridges B_{i,1}, B_{i,2}, \ldots, B_{i,K_i}.
Find the minimum time required to travel from island 1 to island N using each of these bridges at least once.
Only consider the time spent crossing bridges.
You can cross the given bridges in any order and in any direction.

Input

The input is given from Standard Input in the following format:
N M
U_1 V_1 T_1
U_2 V_2 T_2
\vdots
U_M V_M T_M
Q
K_1
B_{1,1} B_{1,2} \cdots B_{1,{K_1}}
K_2
B_{2,1} B_{2,2} \cdots B_{2,{K_2}}
\vdots
K_Q
B_{Q,1} B_{Q,2} \cdots B_{Q,{K_Q}}

Output

Print Q lines. The i-th line (1 \leq i \leq Q) should contain the answer to the i-th query as an integer.

Constraints


```

## 2024-09-07  —  3 problem(s)

### `lcb/abc370_g`

```
We call a positive integer n a good integer if and only if the sum of its positive divisors is divisible by 3.
You are given two positive integers N and M. Find the number, modulo 998244353, of length-M sequences A of positive integers such that the product of the elements in A is a good integer not exceeding N.

Input

The input is given from Standard Input in the following format:
N M

Output

Print the answer.

Constraints


- 1 \leq N \leq 10^{10}
- 1 \leq M \leq 10^5
- N and M are integers.

Sample Input 1

10 1

Sample Output 1

5

There are five sequences that satisfy the conditions:

- (2)
- (5)
- (6)
- (8)
- (10)

```

### `lcb/abc370_e`

```
You are given a sequence A = (A_1, A_2, \dots, A_N) of length N and an integer K.
There are 2^{N-1} ways to divide A into several contiguous subsequences. How many of these divisions have no subsequence whose elements sum to K? Find the count modulo 998244353.
Here, "to divide A into several contiguous subsequences" means the following procedure.

- Freely choose the number k (1 \leq k \leq N) of subsequences and an integer sequence (i_1, i_2, \dots, i_k, i_{k+1}) satisfying 1 = i_1 \lt i_2 \lt \dots \lt i_k \lt i_{k+1} = N+1.
- For each 1 \leq n \leq k, the n-th subsequence is formed by taking the i_n-th through (i_{n+1} - 1)-th elements of A, maintaining their order.

Here are some examples of divisions for A = (1, 2, 3, 4, 5):

- (1, 2, 3), (4), (5)
- (1, 2), (3, 4, 5)
- (1, 2, 3, 4, 5)

Input

The input is given from Standard Input in the following format:
N K
A_1 A_2 \dots A_N

Output

Print the count modulo 998244353.

Constraints


- 1 \leq N \leq 2 \times 10^5
- -10^{15} \leq K \leq 10^{15}
- -10^9 \leq A_i \leq 10^9
- All input values are integers.

Sample Input 1

3 3
1 2 3
```

### `lcb/abc370_f`

```
There is a circular cake divided into N pieces by cut lines. Each cut line is a line segment connecting the center of the circle to a point on the arc.
The pieces and cut lines are numbered 1, 2, \ldots, N in clockwise order, and piece i has a mass of A_i. Piece 1 is also called piece N + 1.
Cut line i is between pieces i and i + 1, and they are arranged clockwise in this order: piece 1, cut line 1, piece 2, cut line 2, \ldots, piece N, cut line N.
We want to divide this cake among K people under the following conditions. Let w_i be the sum of the masses of the pieces received by the i-th person.

- Each person receives one or more consecutive pieces.
- There are no pieces that no one receives.
- Under the above two conditions, \min(w_1, w_2, \ldots, w_K) is maximized.

Find the value of \min(w_1, w_2, \ldots, w_K) in a division that satisfies the conditions, and the number of cut lines that are never cut in the divisions that satisfy the conditions. Here, cut line i is considered cut if pieces i and i + 1 are given to different people.

Input

The input is given from Standard Input in the following format:
N K
A_1 A_2 \ldots A_N

Output

Let x be the value of \min(w_1, w_2, \ldots, w_K) in a division that satisfies the conditions, and y be the number of cut lines that are never cut. Print x and y in this order, separated by a space.

Constraints


- 2 \leq K \leq N \leq 2 \times 10^5
- 1 \leq A_i \leq 10^4
- All input values are integers.

Sample Input 1

5 2
3 6 8 6 4

Sample Output 1

```

## 2024-09-14  —  3 problem(s)

### `lcb/abc371_g`

```
You are given permutations P = (P_1, P_2, \ldots, P_N) and A = (A_1, A_2, \ldots, A_N) of (1,2,\ldots,N).
You can perform the following operation any number of times, possibly zero:

- replace A_i with A_{P_i} simultaneously for all i=1,2,\ldots,N.

Print the lexicographically smallest A that can be obtained.
What is lexicographical order?
 For sequences of length N, A = (A_1, A_2, \ldots, A_N) and B = (B_1, B_2, \ldots, B_N), A is lexicographically smaller than B if and only if:

- there exists an integer i\ (1\leq i\leq N) such that A_i < B_i, and A_j = B_j for all 1\leq j < i.

Input

The input is given from Standard Input in the following format:
N
P_1 P_2 \ldots P_N
A_1 A_2 \ldots A_N

Output

Let (A_1, A_2, \ldots, A_N) be the lexicographically smallest A that can be obtained. Print A_1, A_2, \ldots, A_N in this order, separated by spaces, in one line.

Constraints


- 1\leq N\leq2\times10^5
- 1\leq P_i\leq N\ (1\leq i\leq N)
- P_i\neq P_j\ (1\leq i<j\leq N)
- 1\leq A_i\leq N\ (1\leq i\leq N)
- A_i\neq A_j\ (1\leq i<j\leq N)
- All input values are integers.

Sample Input 1

6
```

### `lcb/abc371_f`

```
There is a road extending east and west, and N persons are on the road.
The road extends infinitely long to the east and west from a point called the origin.
The i-th person (1\leq i\leq N) is initially at a position X_i meters east from the origin.
The persons can move along the road to the east or west.
Specifically, they can perform the following movement any number of times.

- Choose one person. If there is no other person at the destination, move the chosen person 1 meter east or west.

They have Q tasks in total, and the i-th task (1\leq i\leq Q) is as follows.

- The T_i-th person arrives at coordinate G_i.

Find the minimum total number of movements required to complete all Q tasks in order.

Input

The input is given from Standard Input in the following format:
N
X_1 X_2 \ldots X_N
Q
T_1 G_1
T_2 G_2
\vdots
T_Q G_Q

Output

Print the answer.

Constraints


- 1\leq N\leq2\times10^5
- 0\leq X_1 < X_2 < \dotsb < X_N \leq10^8
- 1\leq Q\leq2\times10^5
```

### `lcb/abc371_e`

```
You are given a sequence of integers A = (A_1, A_2, \ldots, A_N) of length N.
                    Define f(l, r) as:

- the number of distinct values in the subsequence (A_l, A_{l+1}, \ldots, A_r).

Evaluate the following expression:

\displaystyle \sum_{i=1}^{N}\sum_{j=i}^N f(i,j).

Input


The input is given from Standard Input in the following format:
N
A_1 \ldots A_N

Output


Print the answer.

Constraints



- 1\leq N\leq 2\times 10^5
- 1\leq A_i\leq N
- All input values are integers.

Sample Input 1


3
1 2 2

```

## 2024-09-21  —  3 problem(s)

### `lcb/abc372_e`

```
There is an undirected graph with N vertices and 0 edges. The vertices are numbered 1 to N.
You are given Q queries to process in order. Each query is of one of the following two types:

- Type 1: Given in the format 1 u v. Add an edge between vertices u and v.
- Type 2: Given in the format 2 v k. Print the k-th largest vertex number among the vertices connected to vertex v. If there are fewer than k vertices connected to v, print -1.

Input

The input is given from Standard Input in the following format:
N Q
\mathrm{query}_1
\mathrm{query}_2
\vdots
\mathrm{query}_Q

Here, \mathrm{query}_i is the i-th query and is given in one of the following formats:
1 u v

2 v k

Output

Let q be the number of Type 2 queries. Print q lines.
The i-th line should contain the answer to the i-th Type 2 query.

Constraints


- 1 \leq N, Q \leq 2 \times 10^5
- In a Type 1 query, 1 \leq u < v \leq N.
- In a Type 2 query, 1 \leq v \leq N, 1 \leq k \leq 10.
- All input values are integers.

Sample Input 1

```

### `lcb/abc372_g`

```
You are given three length-N sequences of positive integers: A=(A_1,A_2,\ldots,A_N), B=(B_1,B_2,\ldots,B_N), and C=(C_1,C_2,\ldots,C_N).  
Find the number of pairs of positive integers (x, y) that satisfy the following condition:  

- A_i \times x + B_i \times y < C_i for all 1 \leq i \leq N.  

It can be proved that the number of such pairs of positive integers satisfying the condition is finite.  
You are given T test cases, each of which should be solved.

Input

The input is given from Standard Input in the following format. Here, \mathrm{case}_i refers to the i-th test case.
T  
\mathrm{case}_1  
\mathrm{case}_2  
\vdots  
\mathrm{case}_T  

Each test case is given in the following format:
N  
A_1 B_1 C_1  
A_2 B_2 C_2  
\vdots  
A_N B_N C_N

Output

Print T lines. The i-th line (1 \leq i \leq T) should contain the answer for \mathrm{case}_i.

Constraints


- 1 \leq T \leq 2 \times 10^5 
- 1 \leq N \leq 2 \times 10^5 
- 1 \leq A_i, B_i, C_i \leq 10^9 
- The sum of N over all test cases is at most 2 \times 10^5.  
```

### `lcb/abc372_f`

```
There is a simple directed graph G with N vertices and N+M edges. The vertices are numbered 1 to N, and the edges are numbered 1 to N+M.
Edge i (1 \leq i \leq N) goes from vertex i to vertex i+1. (Here, vertex N+1 is considered as vertex 1.)
Edge N+i (1 \leq i \leq M) goes from vertex X_i to vertex Y_i.
Takahashi is at vertex 1. At each vertex, he can move to any vertex to which there is an outgoing edge from the current vertex.
Compute the number of ways he can move exactly K times.
That is, find the number of integer sequences (v_0, v_1, \dots, v_K) of length K+1 satisfying all of the following three conditions:

- 1 \leq v_i \leq N for i = 0, 1, \dots, K.
- v_0 = 1.
- There is a directed edge from vertex v_{i-1} to vertex v_i for i = 1, 2, \ldots, K.

Since this number can be very large, print it modulo 998244353.

Input

The input is given from Standard Input in the following format:
N M K
X_1 Y_1
X_2 Y_2
\vdots
X_M Y_M

Output

Print the count modulo 998244353.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 0 \leq M \leq 50
- 1 \leq K \leq 2 \times 10^5
- 1 \leq X_i, Y_i \leq N, X_i \neq Y_i
- All of the N+M directed edges are distinct.
- All input values are integers.
```

## 2024-09-22  —  3 problem(s)

### `lcb/arc184_d`

```
There are N balls on a two-dimensional plane, numbered from 1 to N. Ball i is at point (X_i, Y_i). Here, X = (X_1, X_2, \dots, X_N) and Y = (Y_1, Y_2, \dots, Y_N) are permutations of (1, 2, \dots, N).
You can perform the following operation any number of times:

- Choose one of the remaining balls, say ball k. Then, for each remaining ball i, if either "X_i < X_k and Y_i < Y_k" or "X_i > X_k and Y_i > Y_k" holds, remove ball i.

Find the number of possible sets of balls remaining after performing operations, modulo 998244353.

Input

The input is given from Standard Input in the following format:
N
X_1 Y_1
X_2 Y_2
\vdots
X_N Y_N

Output

Print the answer in one line.

Constraints


- 1 \leq N \leq 300
- X and Y are permutations of (1, 2, \dots, N).

Sample Input 1

3
1 3
2 1
3 2

Sample Output 1

```

### `lcb/arc184_c`

```
We have a long, thin piece of paper whose thickness can be ignored. We perform the following operation 100 times: lift the right end, fold it so that it aligns with the left end using the center as a crease. After completing the 100 folds, we unfold the paper back to its original state. At this point, there are 2^{100} - 1 creases on the paper, and these creases can be classified into two types: mountain folds and valley folds. The figure below represents the state after performing the operation twice, where red solid lines represent mountain folds and red dashed lines represent valley folds.

About mountain and valley folds

- A crease is a mountain fold if it is folded so that the back sides of the paper come together at the crease.
- A crease is a valley fold if it is folded so that the front sides of the paper come together at the crease.



You are given a sequence A = (A_1, A_2, \dots, A_N) of N non-negative integers. Here, 0 = A_1 < A_2 < \dots < A_N \leq 10^{18}.
For each integer i from 1 through 2^{100} - A_N - 1, define f(i) as follows:

- The number of k = 1, 2, \dots, N such that the (i + A_k)-th crease from the left is a mountain fold.

Find the maximum value among f(1), f(2), \dots, f(2^{100} - A_N - 1).

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \cdots A_N

Output

Print the answer in one line.

Constraints


- 1 \leq N \leq 10^3
- 0 = A_1 < A_2 < \dots < A_N \leq 10^{18}

Sample Input 1

4
```

### `lcb/arc184_e`

```
You are given N length-M sequences, where each element is 0 or 1. The i-th sequence is A_i = (A_{i, 1}, A_{i, 2}, \dots, A_{i, M}).
For integers i, j \ (1 \leq i, j \leq N), define f(i, j) as follows:

- 
f(i, j) := The smallest non-negative integer x such that A_i and A_j become identical after performing the following operation x times, or 0 if such x does not exist.

- 
For all integers k \ (1 \leq k \leq M) simultaneously, replace A_{i, k} with \displaystyle \left (\sum_{l=1}^{k} A_{i, l} \right ) \bmod 2.




Find \displaystyle \sum_{i=1}^{N} \sum_{j=i}^{N} f(i, j), modulo 998244353.

Input

The input is given from Standard Input in the following format:
N M
A_{1, 1} A_{1, 2} \cdots A_{1, M}
A_{2, 1} A_{2, 2} \cdots A_{2, M}
\vdots
A_{N, 1} A_{N, 2} \cdots A_{N, M}

Output

Print the answer in one line.

Constraints


- 1 \leq N \times M \leq 10^6
- A_{i, j} \in \{0, 1\}

Sample Input 1

```

## 2024-09-28  —  3 problem(s)

### `lcb/abc373_f`

```
There are N types of items. The i-th type of item has a weight of w_i and a value of v_i. Each type has 10^{10} items available.
Takahashi is going to choose some items and put them into a bag with capacity W. He wants to maximize the value of the selected items while avoiding choosing too many items of the same type. Hence, he defines the happiness of choosing k_i items of type i as k_i v_i - k_i^2. He wants to choose items to maximize the total happiness over all types while keeping the total weight at most W. Calculate the maximum total happiness he can achieve.

Input

The input is given from Standard Input in the following format:
N W
w_1 v_1
w_2 v_2
\vdots
w_N v_N

Output

Print the answer.

Constraints


- 1 \leq N \leq 3000
- 1 \leq W \leq 3000
- 1 \leq w_i \leq W
- 1 \leq v_i \leq 10^9
- All input values are integers.

Sample Input 1

2 10
3 4
3 2

Sample Output 1

5

```

### `lcb/abc373_g`

```
There are 2N points P_1,P_2,\ldots,P_N, Q_1,Q_2,\ldots,Q_N on a two-dimensional plane.
The coordinates of P_i are (A_i, B_i), and the coordinates of Q_i are (C_i, D_i).
No three different points lie on the same straight line.
Determine whether there exists a permutation R = (R_1, R_2, \ldots, R_N) of (1, 2, \ldots, N) that satisfies the following condition. If such an R exists, find one.

- For each integer i from 1 through N, let segment i be the line segment connecting P_i and Q_{R_i}.  Then, segment i and segment j (1 \leq  i < j \leq N) never intersect.

Input

The input is given from Standard Input in the following format:
N
A_1 B_1
A_2 B_2
\vdots 
A_N B_N
C_1 D_1
C_2 D_2
\vdots
C_N D_N

Output

If there is no R satisfying the condition, print -1.
If such an R exists, print R_1, R_2, \ldots, R_N separated by spaces. If there are multiple solutions, you may print any of them.

Constraints


- 1 \leq N \leq 300
- 0 \leq A_i, B_i, C_i, D_i \leq 5000 (1 \leq i \leq N)
- (A_i, B_i) \neq (A_j, B_j) (1 \leq i < j \leq N)
- (C_i, D_i) \neq (C_j, D_j) (1 \leq i < j \leq N)
- (A_i, B_i) \neq (C_j, D_j) (1 \leq i, j \leq N)
- No three different points lie on the same straight line.
- All input values are integers.
```

### `lcb/abc373_e`

```
An election is being held with N candidates numbered 1, 2, \ldots, N. There are K votes, some of which have been counted so far.
Up until now, candidate i has received A_i votes.
After all ballots are counted, candidate i (1 \leq i \leq N) will be elected if and only if the number of candidates who have received more votes than them is less than M.  There may be multiple candidates elected.
For each candidate, find the minimum number of additional votes they need from the remaining ballots to guarantee their victory regardless of how the other candidates receive votes.
Formally, solve the following problem for each i = 1,2,\ldots,N.
Determine if there is a non-negative integer X not exceeding K - \displaystyle{\sum_{i=1}^{N}} A_i satisfying the following condition.  If it exists, find the minimum possible such integer.

- If candidate i receives X additional votes, then candidate i will always be elected.

Input

The input is given from Standard Input in the following format:
N M K
A_1 A_2 \ldots A_N

Output

Let C_i be the minimum number of additional votes candidate i needs from the remaining ballots to guarantee their victory regardless of how other candidates receive votes. Print C_1, C_2, \ldots, C_N separated by spaces.
If candidate i has already secured their victory, then let C_i = 0. If candidate i cannot secure their victory under any circumstances, then let C_i = -1.

Constraints


- 1 \leq M \leq N \leq 2 \times 10^5
- 1 \leq K \leq 10^{12}
- 0 \leq A_i \leq 10^{12}
- \displaystyle{\sum_{i=1}^{N} A_i} \leq K
- All input values are integers.

Sample Input 1

5 2 16
3 1 4 1 5

Sample Output 1
```

## 2024-10-05  —  3 problem(s)

### `lcb/abc374_f`

```
KEYENCE is famous for quick delivery.

In this problem, the calendar proceeds as Day 1, Day 2, Day 3, \dots.
There are orders 1,2,\dots,N, and it is known that order i will be placed on Day T_i.
For these orders, shipping is carried out according to the following rules.

- At most K orders can be shipped together.
- Order i can only be shipped on Day T_i or later.
- Once a shipment is made, the next shipment cannot be made until X days later.
- That is, if a shipment is made on Day a, the next shipment can be made on Day a+X.



For each day that passes from order placement to shipping, dissatisfaction accumulates by 1 per day.
That is, if order i is shipped on Day S_i, the dissatisfaction accumulated for that order is (S_i - T_i).
Find the minimum possible total dissatisfaction accumulated over all orders when you optimally schedule the shipping dates.

Input

The input is given from Standard Input in the following format:
N K X
T_1 T_2 \dots T_N

Output

Print the answer as an integer.

Constraints


- All input values are integers.
- 1 \le K \le N \le 100
- 1 \le X \le 10^9
- 1 \le T_1 \le T_2 \le \dots \le T_N \le 10^{12}

```

### `lcb/abc374_g`

```
All KEYENCE product names consist of two uppercase English letters.
They have already used N product names, the i-th of which (1\leq i\leq N) is S_i.
Once a product name is used, it cannot be reused, so they decided to create an NG (Not Good) list to quickly identify previously used product names.
The NG list must satisfy the following conditions.

- It consists of one or more strings, each consisting of uppercase English letters.
- For each already used product name, there exists at least one string in the list that contains the name as a (contiguous) substring.
- None of the strings in the list contain any length-2 (contiguous) substring that is not an already used product name.

Find the minimum possible number of strings in the NG list.

Input

The input is given from Standard Input in the following format:
N
S_1
S_2
\vdots
S_N

Output

Print the minimum possible number of strings in the NG list.

Constraints


- 1\leq N\leq 26^2
- N is an integer.
- Each S_i is a string of length 2 consisting of uppercase English letters.
- All S_1,S_2,\ldots,S_N are distinct.

Sample Input 1

7
```

### `lcb/abc374_e`

```
The manufacturing of a certain product requires N processes numbered 1,2,\dots,N.
For each process i, there are two types of machines S_i and T_i available for purchase to handle it.

- Machine S_i: Can process A_i products per day per unit, and costs P_i yen per unit.
- Machine T_i: Can process B_i products per day per unit, and costs Q_i yen per unit.

You can purchase any number of each machine, possibly zero.
Suppose that process i can handle W_i products per day as a result of introducing machines.
Here, we define the production capacity as the minimum of W, that is, \displaystyle \min^{N}_{i=1} W_i.
Given a total budget of X yen, find the maximum achievable production capacity.

Input

The input is given from Standard Input in the following format:
N X
A_1 P_1 B_1 Q_1
A_2 P_2 B_2 Q_2
\vdots
A_N P_N B_N Q_N

Output

Print the answer as an integer.

Constraints


- All input values are integers.
- 1 \le N \le 100
- 1 \le A_i,B_i \le 100
- 1 \le P_i,Q_i,X \le 10^7

Sample Input 1

3 22
```

## 2024-10-12  —  3 problem(s)

### `lcb/abc375_e`

```
There are N people divided into three teams.
The people are numbered 1, 2, \ldots, N, and the teams are numbered 1, 2, 3. Currently, person i belongs to team A_i.
Each person has a value called strength; person i has a strength of B_i. The strength of a team is defined as the sum of the strengths of its members.
Determine whether it is possible for zero or more people to switch teams so that all teams have equal strength. If it is possible, find the minimum number of people who need to switch teams to achieve this.
You cannot create new teams other than teams 1, 2, 3.

Input

The input is given from Standard Input in the following format:
N
A_1 B_1
A_2 B_2
\vdots
A_N B_N

Output

If it is possible to make all teams have equal strength, print the minimum number of people who need to switch teams. Otherwise, print -1.

Constraints


- 3 \leq N \leq 100
- A_i \in \lbrace 1, 2, 3 \rbrace
- For each x \in \lbrace 1, 2, 3 \rbrace, there exists some i with A_i = x.
- 1 \leq B_i
- \displaystyle\sum_{i = 1}^{N} B_i \leq 1500 
- All input values are integers.

Sample Input 1

6
1 2
2 5
1 5
```

### `lcb/abc375_g`

```
In the nation of AtCoder, there are N cities numbered 1 to N, and M roads numbered 1 to M.
Road i connects cities A_i and B_i bidirectionally and has a length of C_i.
For each i = 1, \ldots, M, determine whether the following two values are different.

- The shortest distance from city 1 to city N when all roads are passable
- The shortest distance from city 1 to city N when the M - 1 roads other than road i are passable

If city N can be reached from city 1 in one of these cases but not the other, the two values are considered different.

Input

The input is given from Standard Input in the following format:
N M
A_1 B_1 C_1
\vdots
A_M B_M C_M

Output

Print M lines. The i-th line should contain Yes if the shortest distance from city 1 to city N when all roads are passable is different from the shortest distance when the M - 1 roads other than road i are passable, and No otherwise.
If city N can be reached from city 1 in one of these cases but not the other, the two values are considered different.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 1 \leq M \leq 2 \times 10^5
- 1 \leq A_i < B_i \leq N
- All pairs (A_i, B_i) are distinct.
- 1 \leq C_i \leq 10^9
- City N can be reached from city 1 when all roads are passable.
- All input values are integers.

Sample Input 1

```

### `lcb/abc375_f`

```
In the nation of AtCoder, there are N cities numbered 1 to N, and M roads numbered 1 to M.
Road i connects cities A_i and B_i bidirectionally and has a length of C_i.
You are given Q queries to process in order. The queries are of the following two types.

- 1 i: Road i becomes closed.
- 2 x y: Print the shortest distance from city x to city y, using only roads that are not closed. If city y cannot be reached from city x, print -1 instead.

It is guaranteed that each test case contains at most 300 queries of the first type.

Input

The input is given from Standard Input in the following format:
N M Q
A_1 B_1 C_1
\vdots
A_M B_M C_M
\mathrm{query}_1
\vdots
\mathrm{query}_Q

Each query is in one of the following two formats:
1 i

2 x y

Output

Process the queries in order.

Constraints


- 2 \leq N \leq 300
- 0 \leq M \leq \frac{N(N-1)}{2}
- 1 \leq A_i < B_i \leq N
```

## 2024-10-13  —  4 problem(s)

### `lcb/arc185_c`

```
You are given an integer sequence A = (A_1, A_2, \dots, A_N) and an integer X.
Print one triple of integers (i, j, k) satisfying all of the following conditions. If no such triple exists, report that fact.

- 1 \leq i \lt j \lt k \leq N
- A_i + A_j + A_k = X

Input

The input is given from Standard Input in the following format:
N X
A_1 A_2 \dots A_N

Output

If there exists an integer triple (i, j, k) satisfying the conditions, print one in the following format. If there are multiple solutions, you may print any of them.
i j k

If no such triple exists, print -1.

Constraints


- 3 \leq N \leq 10^6
- 1 \leq X \leq 10^6
- 1 \leq A_i \leq X
- All input values are integers.

Sample Input 1

5 16
1 8 5 10 13

Sample Output 1

1 3 4
```

### `lcb/arc185_e`

```
Define the score of a sequence of positive integers B = (B_1, B_2, \dots, B_k) as \displaystyle \sum_{i=1}^{k-1} \gcd(B_i, B_{i+1}).
Given a sequence of positive integers A = (A_1, A_2, \dots, A_N), solve the following problem for m = 1, 2, \dots, N.

- There are 2^m - 1 non-empty subsequences of the sequence (A_1, A_2, \dots, A_m). Find the sum of the scores of all those subsequences, modulo 998244353. Two subsequences are distinguished if they are taken from different positions in the sequence, even if they coincide as sequences.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \dots A_N

Output

Print N lines. The i-th line should contain the answer for m = i.

Constraints


- 1 \leq N \leq 5 \times 10^5
- 1 \leq A_i \leq 10^5
- All input values are integers.

Sample Input 1

3
9 6 4

Sample Output 1

0
3
11

Consider the case m = 3. Here are the non-empty subsequences of (A_1, A_2, A_3) = (9, 6, 4) and their scores.

```

### `lcb/arc185_b`

```
You are given an integer sequence A = (A_1, A_2, \dots, A_N) of length N.
You can perform the following operation any number of times, possibly zero:

- Choose an integer pair (i, j) satisfying 1 \leq i \lt j \leq N, and replace A_i with A_i + 1 and A_j with A_j - 1.

Determine whether it is possible to make A a non-decreasing sequence through the operations.
You are given T test cases. Solve each of them.

Input

The input is given from Standard Input in the following format. Here, \mathrm{case}_i denotes the i-th test case.
T
\mathrm{case}_1
\mathrm{case}_2
\vdots
\mathrm{case}_T

Each test case is given in the following format:
N
A_1 A_2 \dots A_N

Output

Print T lines. The i-th line should contain the answer for the i-th test case.
For each test case, if it is possible to make A a non-decreasing sequence through the operations, print Yes; otherwise, print No.

Constraints


- 1 \leq T \leq 2 \times 10^5
- 2 \leq N \leq 2 \times 10^5
- 0 \leq A_i \leq 10^9
- The sum of N over all test cases is at most 2 \times 10^5.
- All input values are integers.

```

### `lcb/arc185_d`

```
There is a tree with N \times M + 1 vertices numbered 0, 1, \dots, N \times M. The i-th edge (1 \leq i \leq N \times M) connects vertices i and \max(i - N, 0).
Vertex 0 is painted. The other vertices are unpainted.
Takahashi is at vertex 0. As long as there exists an unpainted vertex, he performs the following operation:

- He chooses one of the vertices adjacent to his current vertex uniformly at random (all choices are independent) and moves to that vertex. Then, if the vertex he is on is unpainted, he paints it.

Find the expected number of times he performs the operation, modulo 998244353.

What is the expected value modulo 998244353?

It can be proved that the sought expected value is always rational. Under the constraints of this problem, when that value is expressed as an irreducible fraction \frac{P}{Q}, it can also be proved that Q \not\equiv 0 \pmod{998244353}. Then, there uniquely exists an integer R such that R \times Q \equiv P \pmod{998244353}, 0 \leq R \lt 998244353. Report this R.

Input

The input is given from Standard Input in the following format:
N M

Output

Print the expected number of times he performs the operation, modulo 998244353.

Constraints


- 1 \leq N \leq 2 \times 10^5
- 1 \leq M \leq 2 \times 10^5
- N and M are integers.

Sample Input 1

2 2

Sample Output 1

20
```

## 2024-10-19  —  3 problem(s)

### `lcb/abc376_g`

```
There is a rooted tree with N + 1 vertices numbered from 0 to N. Vertex 0 is the root, and the parent of vertex i is vertex p_i.
One of the vertices among vertex 1, vertex 2, ..., vertex N hides a treasure. The probability that the treasure is at vertex i is \frac{a_i}{\sum_{j=1}^N a_j}.
Also, each vertex is in one of the two states: "searched" and "unsearched". Initially, vertex 0 is searched, and all other vertices are unsearched.
Until the vertex containing the treasure becomes searched, you perform the following operation:

- Choose an unsearched vertex whose parent is searched, and mark it as searched.

Find the expected number of operations required when you act to minimize the expected number of operations, modulo 998244353.
You are given T test cases; solve each of them.

How to find an expected value modulo 998244353

It can be proved that the expected value is always a rational number. Under the constraints of this problem, it can also be proved that when the expected value is expressed as an irreducible fraction \frac{P}{Q}, we have Q \not\equiv 0 \pmod{998244353}. In this case, there is a unique integer R satisfying R \times Q \equiv P \pmod{998244353},\ 0 \leq R < 998244353. Report this R.

Input

The input is given from Standard Input in the following format. Here, \mathrm{case}_i denotes the i-th test case.
T
\mathrm{case}_1
\mathrm{case}_2
\vdots
\mathrm{case}_T

Each test case is given in the following format:
N
p_1 p_2 \dots p_N
a_1 a_2 \dots a_N

Output

Print T lines. The i-th line should contain the answer for the i-th test case.

Constraints


```

### `lcb/abc376_f`

```
Note: This problem has almost the same setting as Problem B. Only the parts in bold in the main text and constraints differ.
You are holding a ring with both hands.
This ring consists of N\ (N \geq 3) parts numbered 1,2,\dots,N, where parts i and i+1 (1 \leq i \leq N-1) are adjacent, and parts 1 and N are also adjacent.
Initially, your left hand is holding part 1, and your right hand is holding part 2.
In one operation, you can do the following:

- Move one of your hands to an adjacent part of the part it is currently holding. However, you can do this only if the other hand is not on the destination part.

The following figure shows the initial state and examples of operations that can and cannot be made from there. The number written on each part of the ring represents the part number, and the circles labeled L and R represent your left and right hands, respectively.

You need to follow Q instructions given to you in order.
The i-th (1 \leq i \leq Q) instruction is represented by a character H_i and an integer T_i, meaning the following:

- Perform some number of operations (possibly zero) so that your left hand (if H_i is L) or your right hand (if H_i is R) is holding part T_i.
  Here, you may move the other hand not specified by H_i.

Under the settings and constraints of this problem, it can be proved that any instructions are achievable.
Find the minimum total number of operations required to follow all the instructions.

Input

The Input is given from Standard Input in the following format:
N Q
H_1 T_1
H_2 T_2
\vdots
H_Q T_Q

Output

Print the minimum total number of operations required to follow all the instructions.

Constraints


```

### `lcb/abc376_e`

```
You are given sequences of length N: A = (A_1, A_2, \dots, A_N) and B = (B_1, B_2, \dots, B_N).
Let S be a subset of \lbrace1, 2, \dots, N\rbrace of size K.
Here, find the minimum possible value of the following expression:

\displaystyle \left(\max_{i \in S} A_i\right) \times \left(\sum_{i \in S} B_i\right).

You are given T test cases; solve each of them.

Input

The input is given from Standard Input in the following format. Here, \mathrm{case}_i denotes the i-th test case.
T
\mathrm{case}_1
\mathrm{case}_2
\vdots
\mathrm{case}_T

Each test case is given in the following format:
N K
A_1 A_2 \dots A_N
B_1 B_2 \dots B_N

Output

Print T lines. The i-th line should contain the answer for the i-th test case.

Constraints


- 1 \leq T \leq 2 \times 10^5
- 1 \leq K \leq N \leq 2 \times 10^5
- 1 \leq A_i, B_i \leq 10^6
- The sum of N over all test cases is at most 2 \times 10^5.
- All input values are integers.

```

## 2024-10-26  —  3 problem(s)

### `lcb/abc377_g`

```
You are given N strings S_1,S_2,\ldots,S_N. Each string consists of lowercase English letters.
For each k=1,2,\ldots,N, solve the following problem.

Let T=S_k and consider performing the following two types of operations any number of times in any order:

- Pay a cost of 1 to delete the last character of T. This operation is possible when T is not empty.
- Pay a cost of 1 to add any lowercase English letter to the end of T.

Find the minimum total cost needed to make T either empty or match one of S_1,S_2,\ldots,S_{k-1}.

Input

The input is given from Standard Input in the following format:
N
S_1
S_2
\vdots
S_N

Output

Print N lines.
The i-th line (1\le i\le N) should contain the answer for k=i.

Constraints


- 1\le N\le 2\times 10^5
- Each S_i is a string of length at least 1 consisting of lowercase English letters.
- \displaystyle \sum_{i=1}^N |S_i|\le 2\times 10^5

Sample Input 1

3
snuke
```

### `lcb/abc377_f`

```
There is a grid of N^2 squares with N rows and N columns.
Let (i,j) denote the square at the i-th row from the top (1\leq i\leq N) and j-th column from the left (1\leq j\leq N).
Each square is either empty or has a piece placed on it.
There are M pieces placed on the grid, and the k-th (1\leq k\leq M) piece is placed on square (a_k,b_k).
You want to place your piece on an empty square in such a way that it cannot be captured by any of the existing pieces.
A piece placed on square (i,j) can capture pieces that satisfy any of the following conditions:

- Placed in row i
- Placed in column j
- Placed on any square (a,b)\ (1\leq a\leq N,1\leq b\leq N) where i+j=a+b
- Placed on any square (a,b)\ (1\leq a\leq N,1\leq b\leq N) where i-j=a-b

For example, a piece placed on square (4,4) can capture pieces placed on the squares shown in blue in the following figure:

How many squares can you place your piece on?

Input

The input is given from Standard Input in the following format:
N M
a_1 b_1
a_2 b_2
\vdots
a_M b_M

Output

Print the number of empty squares where you can place your piece without it being captured by any existing pieces.

Constraints


- 1\leq N\leq10^9
- 1\leq M\leq10^3
- 1\leq a_k\leq N,1\leq b_k\leq N\ (1\leq k\leq M)
```

### `lcb/abc377_e`

```
You are given a permutation P=(P_1,P_2,\ldots,P_N) of (1,2,\ldots,N).
The following operation will be performed K times:

- For i=1,2,\ldots,N, simultaneously update P_i to P_{P_i}.

Print P after all operations.

Input

The input is given from Standard Input in the following format:
N K
P_1 P_2 \ldots P_N

Output

For the P after all operations, print P_1,P_2,\ldots,P_N in this order, separated by spaces.

Constraints


- 1\leq N\leq2\times10^5
- 1\leq K\leq10^{18}
- 1\leq P_i\leq N\ (1\leq i\leq N)
- P_i\neq P_j\ (1\leq i\lt j\leq N)
- All input values are integers.

Sample Input 1

6 3
5 6 3 1 2 4

Sample Output 1

6 1 3 2 4 5

```

## 2024-10-27  —  5 problem(s)

### `lcb/arc186_a`

```
For two N \times N matrices A and B whose elements are 0 or 1, we say that A and B are similar if they satisfy the following conditions:

- The sums of corresponding rows are equal. That is, A_{i,1} + \dots + A_{i,N} = B_{i,1} + \dots + B_{i,N} for any i=1,\dots,N.
- The sums of corresponding columns are equal. That is, A_{1,j} + \dots + A_{N,j} = B_{1,j} + \dots + B_{N,j} for any j=1,\dots,N.

Furthermore, for an N \times N matrix A whose elements are 0 or 1, and integers i,j (1 \leq i,j \leq N), we say that the element at row i column j is fixed if A_{i,j} = B_{i,j} holds for any matrix B that is similar to A.
Answer the following Q queries:

- The i-th query: If there exists an N \times N matrix whose elements are 0 or 1 such that exactly K_i elements are fixed, output Yes; otherwise, output No.

Input

The input is given from Standard Input in the following format:
N Q
K_1
K_2
\vdots
K_Q

Output

Output Q lines.
For the i-th line (1 \le i \le Q), output the answer for the i-th query.

Constraints


- 2 \le N \le 30
- 1 \le Q \le N^2+1
- 0 \le K_i \le N^2
- K_i \ne K_j (1 \le i < j \le Q)
- All inputs are integers

Sample Input 1

```

### `lcb/arc186_d`

```
Whether a non-empty sequence of non-negative integers (V_1, V_2, \dots, V_M) is Polish or not is recursively defined as follows:

- We say (V_1, V_2, \dots, V_M) is Polish if there exist V_1 Polish sequences W_1, W_2, \dots, W_{V_1} such that the concatenation of sequences (V_1), W_1, W_2, \dots, W_{V_1} in this order equals (V_1, V_2, \dots, V_M).

In particular, the sequence (0) is Polish.
Given a sequence of non-negative integers (A_1, A_2, \dots, A_N) of length N, find the number of Polish sequences of length N that are lexicographically not greater than (A_1, A_2, \dots, A_N), modulo 998244353.
 What is lexicographical order on sequences?
We say that sequence S = (S_1,S_2,\ldots,S_{|S|}) is lexicographically less than sequence T = (T_1,T_2,\ldots,T_{|T|}) if either condition 1. or 2. below holds.
Here, |S|, |T| represent the lengths of S, T respectively.

-  |S| \lt |T| and (S_1,S_2,\ldots,S_{|S|}) = (T_1,T_2,\ldots,T_{|S|}). 
-  There exists an integer 1 \leq i \leq \min\lbrace |S|, |T| \rbrace such that both of the following hold:

-  (S_1,S_2,\ldots,S_{i-1}) = (T_1,T_2,\ldots,T_{i-1})
-  S_i is (numerically) less than T_i.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \dots A_N

Output

Print the number of sequences satisfying the conditions, modulo 998244353.

Constraints


- 1\leq N \leq 3\times 10^5
- 0\leq A_i \lt N
- All input values are integers.

Sample Input 1

```

### `lcb/arc186_b`

```
You are given a sequence of integers (A_1,\dots,A_N) of length N. This sequence satisfies 0\le A_i < i for each i=1,\dots,N.
Find the number of permutations (P_1,\dots,P_N) of (1,\dots,N) that satisfy the following conditions, modulo 998244353.

- For each i=1,\dots,N:
- P_j > P_i for any integer j with A_i < j < i 
- P_{A_i} < P_i if A_i > 0



For the sequence (A_1,\dots,A_N) given in the input, it is guaranteed that there exists a permutation satisfying the conditions.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \dots A_N

Output

Print the number of permutations satisfying the conditions, modulo 998244353.

Constraints


- 1\le N\le 3\times 10^5
- 0\le A_i \lt i
- For A_1,\dots,A_N, there exists a permutation satisfying the conditions in the problem statement.
- All input values are integers.

Sample Input 1

4
0 1 0 3

Sample Output 1
```

### `lcb/arc186_e`

```
You are given a sequence of integers (X_1,\dots,X_M) of length M consisting of 1,\dots,K.
Find the number of sequences (A_1,\dots,A_N) of length N consisting of 1,\dots,K that satisfy the following condition, modulo 998244353:

- Among all sequences of length M consisting of 1,\dots,K, the only sequence that cannot be obtained as a (not necessarily contiguous) subsequence of (A_1,\dots,A_N) is (X_1,\dots,X_M).

Input

The input is given from Standard Input in the following format:
N M K
X_1 X_2 \dots X_M

Output

Print the number of sequences satisfying the condition, modulo 998244353.

Constraints


- 2\le M,K \le N \le 400
- 1\le X_i \le K
- All input values are integers.

Sample Input 1

5 2 3
1 1

Sample Output 1

4

The following four sequences satisfy the condition:

- (2, 3, 1, 2, 3)
- (2, 3, 1, 3, 2)
```

### `lcb/arc186_c`

```
Mr. Ball and Mr. Box will play a game with balls and boxes.
Initially, Mr. Ball has 10^{100} balls of each of M different types, and Mr. Box has 10^{100} yen.
There are N boxes, where the i-th box has capacity V_i and costs P_i yen. During the game, Mr. Box can buy any box at any time.
In this game, the following operations are repeated until the game ends:

- Mr. Ball chooses one ball and gives it to Mr. Box.
- Mr. Box either accepts the ball or ends the game without accepting it.
- If Mr. Box accepts the ball, he chooses one of his purchased boxes and puts the ball in it.
- If the box with the ball satisfies the following conditions, Mr. Box receives 1 yen. Otherwise, the game ends.
- The number of balls in the box does not exceed its capacity.
- All balls in the box are of the same type.



Mr. Ball will play optimally to minimize Mr. Box's final money, while Mr. Box will play optimally to maximize it.
How much will Mr. Box's money increase throughout the game?
Here, both players have access to all information. In particular, Mr. Ball can see the capacity, price, and contents (type and number of balls) of each box.
Also, note that Mr. Box's initial money is large enough that he will never run out of money to buy boxes.
Solve T test cases for each input file.

Input

The input is given from Standard Input in the following format, where \mathrm{case}_i represents the i-th test case:
T
\mathrm{case}_1
\mathrm{case}_2
\vdots
\mathrm{case}_T

Each test case is given in the following format:
N M
V_1 P_1
V_2 P_2
\vdots
V_N P_N
```

## 2024-11-02  —  4 problem(s)

### `lcb/abc378_e`

```
You are given a sequence A = (A_1, A_2, \dots, A_N) of N non-negative integers, and a positive integer M.
Find the following value:
\[
\sum_{1 \leq l \leq r \leq N} \left( \left(\sum_{l \leq i \leq r} A_i\right) \mathbin{\mathrm{mod}} M \right).
\]
Here, X \mathbin{\mathrm{mod}} M denotes the remainder when the non-negative integer X is divided by M.

Input

The input is given from Standard Input in the following format:
N M
A_1 A_2 \dots A_N

Output

Print the answer.

Constraints


- 1 \leq N \leq 2 \times 10^5
- 1 \leq M \leq 2 \times 10^5
- 0 \leq A_i \leq 10^9

Sample Input 1

3 4
2 5 0

Sample Output 1

10


- A_1 \mathbin{\mathrm{mod}} M = 2
```

### `lcb/abc378_d`

```
There is a grid of H \times W cells. Let (i, j) denote the cell at the i-th row from the top and the j-th column from the left.
Cell (i, j) is empty if S_{i,j} is ., and blocked if it is #.
Count the number of ways to start from an empty cell and make K moves to adjacent cells (up, down, left, or right), without passing through blocked squares and not visiting the same cell more than once.
Specifically, count the number of sequences of length K+1, ((i_0, j_0), (i_1, j_1), \dots, (i_K, j_K)), satisfying the following.

- 1 \leq i_k \leq H, 1 \leq j_k \leq W, and S_{i_k, j_k} is ., for each 0 \leq k \leq K.
- |i_{k+1} - i_k| + |j_{k+1} - j_k| = 1 for each 0 \leq k \leq K-1.
- (i_k, j_k) \neq (i_l, j_l) for each 0 \leq k < l \leq K.

Input

The input is given from Standard Input in the following format:
H W K
S_{1,1}S_{1,2}\dots S_{1,W}
S_{2,1}S_{2,2}\dots S_{2,W}
\vdots
S_{H,1}S_{H,2}\dots S_{H,W}

Output

Print the answer.

Constraints


- 1 \leq H, W \leq 10
- 1 \leq K \leq 11
- H, W, and K are integers.
- Each S_{i,j} is . or #.
- There is at least one empty cell.

Sample Input 1

2 2 2
.#
```

### `lcb/abc378_f`

```
You are given a tree with N vertices. The i-th edge (1 \leq i \leq N-1) connects vertices u_i and v_i bidirectionally.
Adding one undirected edge to the given tree always yields a graph with exactly one cycle.
Among such graphs, how many satisfy all of the following conditions?

- The graph is simple.
- All vertices in the cycle have degree 3.

Input

The input is given from Standard Input in the following format:
N
u_1 v_1
u_2 v_2
\vdots
u_{N-1} v_{N-1}

Output

Print the answer.

Constraints


- 3 \leq N \leq 2 \times 10^5
- 1 \leq u_i, v_i \leq N
- The given graph is a tree.
- All input values are integers.

Sample Input 1

6
1 2
2 3
3 4
4 5
```

### `lcb/abc378_g`

```
You are given integers A, B, and M.
How many permutations P = (P_1, \dots, P_{AB-1}) of (1, 2, \ldots, AB - 1) satisfy all of the following conditions? Find the count modulo M.

- The length of a longest increasing subsequence of P is A.
- The length of a longest decreasing subsequence of P is B.
- There exists an integer n such that appending n + 0.5 to the end of P does not change either of the lengths of a longest increasing subsequence and a longest decreasing subsequence.

Input

The input is given from Standard Input in the following format:
A B M

Output

Print the number of permutations satisfying the conditions, modulo M.

Constraints


- All input values are integers.
- 2 \leq A, B
- AB \leq 120
- 10^8 \leq M \leq 10^9
- M is a prime.

Sample Input 1

3 2 998244353

Sample Output 1

10

For example, P = (2, 4, 5, 1, 3) satisfies the conditions. This can be confirmed as follows:

```

## 2024-11-09  —  3 problem(s)

### `lcb/abc379_e`

```
You are given a string S of length N consisting of digits from 1 through 9.
For each pair of integers (i,j) \ (1\leq i\leq j\leq N), define f(i, j) as the value obtained by interpreting the substring of S from the i-th through the j-th character as a decimal integer. Find \displaystyle \sum_{i=1}^N \sum_{j=i}^N f(i, j).

Input

The input is given from Standard Input in the following format:
N
S

Output

Print the answer.

Constraints


- 1 \leq N \leq 2 \times 10^5
- N is an integer.
- S is a string of length N consisting of digits from 1 through 9.

Sample Input 1

3
379

Sample Output 1

514

The answer is f(1,1) + f(1,2) + f(1,3) + f(2,2) + f(2,3) + f(3,3) = 3 + 37 + 379 + 7 + 79 + 9 = 514.

Sample Input 2

30
314159265358979323846264338327
```

### `lcb/abc379_g`

```
You are given a grid S with H rows and W columns consisting of 1, 2, 3, and ?. The character at the i-th row and j-th column is S_{i,j}.
By replacing each ? in S with 1, 2, or 3, we can obtain 3^q different grids, where q is the number of ?. Among these grids, how many satisfy the following condition? Print the count modulo 998244353.

- Any two adjacent (edge-sharing) cells contain different digits.

Input

The input is given from Standard Input in the following format:
H W
S_{1,1}S_{1,2}\ldots S_{1,W}
S_{2,1}S_{2,2}\ldots S_{2,W}
\vdots
S_{H,1}S_{H,2}\ldots S_{H,W}

Output

Print the answer.

Constraints


- 1 \leq H, W
- H \times W \leq 200
- H and W are integers.
- S is a grid with H rows and W columns consisting of 1, 2, 3, and ?.

Sample Input 1

2 2
1?
??

Sample Output 1

6
```

### `lcb/abc379_f`

```
There are N buildings, building 1, building 2, \ldots, building N, arranged in this order in a straight line from west to east. Building 1 is the westernmost, and building N is the easternmost. The height of building i\ (1\leq i\leq N) is H_i.
For a pair of integers (i,j)\ (1\leq i\lt j\leq N), building j can be seen from building i if the following condition is satisfied.

- There is no building taller than building j between buildings i and j. In other words, there is no integer k\ (i\lt k\lt j) such that H_k > H_j.

You are given Q queries. In the i-th query, given a pair of integers (l_i,r_i)\ (l_i\lt r_i), find the number of buildings to the east of building r_i (that is, buildings r_i + 1, r_i + 2, \ldots, N) that can be seen from both buildings l_i and r_i.

Input

The input is given from Standard Input in the following format:
N Q
H_1 H_2 \ldots H_N
l_1 r_1
l_2 r_2
\vdots
l_Q r_Q

Output

Print Q lines. The i-th line (1 \leq i \leq Q) should contain the answer to the i-th query.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 1 \leq Q \leq 2 \times 10^5
- 1 \leq H_i \leq N
- H_i\neq H_j\ (i\neq j)
- 1 \leq l_i < r_i \leq N
- All input values are integers.

Sample Input 1

5 3
2 1 4 3 5
```

## 2024-11-16  —  3 problem(s)

### `lcb/abc380_e`

```
There are N cells in a row, numbered 1 to N.
For each 1 \leq i < N, cells i and i+1 are adjacent.
Initially, cell i is painted with color i.
You are given Q queries. Process them in order. Each query is of one of the following two types.

- 1 x c: Repaint the following to color c: all reachable cells reachable from cell x by repeatedly moving to an adjacent cell painted in the same color as the current cell.
- 2 c: Print the number of cells painted with color c.

Input

The input is given from Standard Input in the following format:
N Q
\mathrm{query}_1
\vdots
\mathrm{query}_Q

Each query is given in one of the following two formats:
1 x c

2 c

Output

Let q be the number of queries of the second type. Print q lines.
The i-th line should contain the answer to the i-th such query.

Constraints


- 1 \leq N \leq 5 \times 10^5
- 1 \leq Q \leq 2 \times 10^5
- In queries of the first type, 1 \leq x \leq N.
- In queries of the first and second types, 1 \leq c \leq N.
- There is at least one query of the second type.
- All input values are integers.
```

### `lcb/abc380_f`

```
Takahashi and Aoki will play a game using cards with numbers written on them.
Initially, Takahashi has N cards with numbers A_1, \ldots, A_N in his hand, Aoki has M cards with numbers B_1, \ldots, B_M in his hand, and there are L cards with numbers C_1, \ldots, C_L on the table.
Throughout the game, both Takahashi and Aoki know all the numbers on all the cards, including the opponent's hand.
Starting with Takahashi, they take turns performing the following action:

- Choose one card from his hand and put it on the table. Then, if there is a card on the table with a number less than the number on the card he just played, he may take one such card from the table into his hand.

The player who cannot make a move first loses, and the other player wins. Determine who wins if both players play optimally.
It can be proved that the game always ends in a finite number of moves.

Input

The input is given from Standard Input in the following format:
N M L
A_1 \ldots A_N
B_1 \ldots B_M
C_1 \ldots C_L

Output

Print Takahashi if Takahashi wins, and Aoki if Aoki wins.

Constraints


- 1 \leq N, M, L
- N + M + L \leq 12
- 1 \leq A_i, B_i, C_i \leq 10^9
- All input values are integers.

Sample Input 1

1 1 2
2
4
```

### `lcb/abc380_g`

```
You are given a permutation P of (1,2,\dots,N) and an integer K.  
Find the expected value, modulo 998244353, of the inversion number of P after performing the following operation:

- First, choose an integer i uniformly at random between 1 and N - K + 1, inclusive.
- Then, shuffle P_i, P_{i+1}, \dots, P_{i+K-1} uniformly at random.


What is the inversion number?
The inversion number of a sequence (A_1, A_2, \dots, A_N) is the number of integer pairs (i, j) satisfying 1 \le i < j \le N and A_i > A_j.


What does "expected value modulo 998244353" mean?
It can be proved that the sought expected value is always rational. Under the constraints of this problem, when this value is represented as an irreducible fraction \frac{P}{Q}, it can also be proved that Q \not\equiv 0 \pmod{998244353}. Thus, there is a unique integer R satisfying R \times Q \equiv P \pmod{998244353}, \ 0 \le R < 998244353. Report this integer R.

Input

The input is given from Standard Input in the following format:
N K
P_1 P_2 \dots P_N

Output

Print the answer in one line.

Constraints


- All input values are integers.
- 1 \le K \le N \le 2 \times 10^5
- P is a permutation of (1,2,\dots,N).

Sample Input 1

4 2
1 4 2 3
```

## 2024-11-17  —  1 problem(s)

### `lcb/arc187_b`

```
For a sequence A = (A_1, \ldots, A_N) of length N, define f(A) as follows.

- Prepare a graph with N vertices labeled 1 to N and zero edges. For every integer pair (i, j) satisfying 1 \leq i < j \leq N, if A_i \leq A_j, draw a bidirectional edge connecting vertices i and j. Define f(A) as the number of connected components in the resulting graph.

You are given a sequence B = (B_1, \ldots, B_N) of length N. Each element of B is -1 or an integer between 1 and M, inclusive.
By replacing every occurrence of -1 in B with an integer between 1 and M, one can obtain M^q sequences B', where q is the number of -1 in B.
Find the sum, modulo 998244353, of f(B') over all possible B'.

Input

The input is given from Standard Input in the following format:
N M
B_1 \ldots B_N

Output

Print the answer.

Constraints


- All input numbers are integers.
- 2 \leq N \leq 2000
- 1 \leq M \leq 2000
- Each B_i is -1 or an integer between 1 and M, inclusive.

Sample Input 1

3 3
2 -1 1

Sample Output 1

6

```

## 2024-11-22  —  2 problem(s)

### `lcb/abc381_e`

```
The definition of an 11/22 string in this problem is the same as in Problems A and C.

A string T is called an 11/22 string when it satisfies all of the following conditions:

- |T| is odd. Here, |T| denotes the length of T.
- The 1-st through (\frac{|T|+1}{2} - 1)-th characters are all 1.
- The (\frac{|T|+1}{2})-th character is /.
- The (\frac{|T|+1}{2} + 1)-th through |T|-th characters are all 2.

For example, 11/22, 111/222, and / are 11/22 strings, but 1122, 1/22, 11/2222, 22/11, and //2/2/211 are not.
Given a string S of length N consisting of 1, 2, and /, process Q queries.
Each query provides two integers L and R. Let T be the (contiguous) substring of S from the L-th through R-th character. Find the maximum length of a subsequence (not necessarily contiguous) of T that is an 11/22 string. If no such subsequence exists, print 0.

Input

The input is given from Standard Input in the following format. Here, \mathrm{query}_i denotes the i-th query.
N Q
S
\mathrm{query}_1
\mathrm{query}_2
\vdots
\mathrm{query}_Q

Each query is given in the following format:
L R

Output

Print Q lines. The i-th line should contain the answer to the i-th query.

Constraints


- 1 \leq N \leq 10^5
- 1 \leq Q \leq 10^5
```

### `lcb/abc381_d`

```
A sequence X = (X_1, X_2, \ldots) of positive integers (possibly empty) is called a 1122 sequence if and only if it satisfies all of the following three conditions: (The definition of a 1122 sequence is the same as in Problem F.)

- \lvert X \rvert is even. Here, \lvert X \rvert denotes the length of X.
- For each integer i satisfying 1\leq i\leq \frac{|X|}{2}, X_{2i-1} and X_{2i} are equal.
- Each positive integer appears in X either not at all or exactly twice. That is, every positive integer contained in X appears exactly twice in X.

Given a sequence A = (A_1, A_2, \ldots, A_N) of length N consisting of positive integers, print the maximum length of a (contiguous) subarray of A that is a 1122 sequence.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \ldots A_N

Output

Print the maximum length of a (contiguous) subarray of A that is a 1122 sequence.

Constraints


- 1\leq N \leq 2 \times 10^5
- 1\leq A_i \leq N
- All input values are integers.

Sample Input 1

8
2 3 1 1 2 2 1 1

Sample Output 1

4

For example, taking the subarray from the 3-rd to 6-th elements of A, we get (1, 1, 2, 2), which is a 1122 sequence of length 4.
```

## 2024-11-23  —  4 problem(s)

### `lcb/arc188_c`

```
There is a village with N villagers numbered from 1 to N.
Each villager is honest or a liar. Additionally, some villagers are confused.
You have obtained M testimonies from the villagers. Each testimony is given by A_i, B_i, C_i for i=1,2,\ldots,M, representing:

- If C_i=0, villager A_i testified that villager B_i is honest.
- If C_i=1, villager A_i testified that villager B_i is a liar.

All villagers know whether every other villager is honest or a liar, and you know that they made their testimonies to you according to the following rules:

- An honest villager who is not confused always tells the truth.
- A liar who is not confused always tells lies.
- A confused honest villager always tells lies.
- A confused liar always tells the truth.

In other words, if they are not confused, honest villagers always tell the truth, and liars always tell lies, but if they are confused, it is reversed.
You have decided to guess the set of villagers who are confused.
Given a choice of villagers who are confused, whether the set of testimonies "contradicts" or not is determined.
Here, a set of testimonies is said to contradict if, no matter how you assign honest or liar statuses to the villagers, there is at least one testimony that violates the villagers' testimony rules.
Find a set of confused villagers such that the given set of testimonies does not contradict.
If no such set of confused villagers exists, indicate that fact.

Input

The input is given from Standard Input in the following format:
N M
A_1 B_1 C_1
A_2 B_2 C_2
\vdots
A_M B_M C_M

Output

If there exists a set of confused villagers such that the given set of testimonies does not contradict, print a string of length N representing the set of confused villagers. In this string, the i-th character should be 1 if villager i is confused, and 0 otherwise.
If no such set of confused villagers exists, print -1.

```

### `lcb/arc188_b`

```
On a circle, there are N equally spaced points numbered 0,1,\ldots,N-1 in this order, with Alice at point 0 and Bob at point K. Initially, all points are colored white. Starting with Alice, they alternately perform the following operation:

- Choose one of the currently white points and color it black. Here, after the operation, the coloring of the points must be symmetric with respect to the straight line connecting the operator and the center of the circle.

If the operator cannot perform an operation satisfying the above condition, the sequence of operations ends there.
Both players cooperate and make the best choices to maximize the total number of points colored black in the end. Determine whether all points are colored black at the end of the sequence of operations.
You are given T test cases to solve.

Input

The input is given from Standard Input in the following format:
T
\mathrm{case}_1
\mathrm{case}_2
\vdots 
\mathrm{case}_T

Each test case \mathrm{case}_i (1 \leq i \leq T) is in the following format:
N K

Output

Print T lines. The i-th line should contain Yes if all points can be colored black for the i-th test case, and No otherwise.

Constraints


- 1 \leq T \leq 10^5
- 2 \leq N \leq 2 \times 10^5
- 1 \leq K \leq N-1
- All input values are integers.

Sample Input 1

4
```

### `lcb/arc188_d`

```
You are going to create N sequences of length 3, satisfying the following conditions.

- For each of k = 1,2,3, the following holds:
- Among the k-th elements of the sequences, each integer from 1 through N appears exactly once.



For this sequence of sequences, define sequences a=(a_1,a_2,\ldots,a_N) and b=(b_1,b_2,\ldots,b_N) as follows.

- Let s_i be the i-th sequence, and let t_i be the reverse of the i-th sequence. When all of these are sorted in lexicographical order, s_i comes a_i-th, and t_i comes b_i-th.
- Here, if there are identical sequences among the 2N sequences, a and b are not defined.

Therefore, if a and b are defined, each integer from 1 through 2N appears exactly once in the concatenation of a and b.
You are given sequences A and B of length N, where each element of A is an integer between 1 and 2N, and each element of B is either an integer between 1 and 2N or -1.
Also, in the concatenation of A and B, each integer other than -1 appears at most once.
How many pairs of sequences a,b are there such that a and b are defined and the following holds for each integer i from 1 through N?

- a_i = A_i.
- b_i = B_i if B_i \neq -1.

Find the count modulo 998244353.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \ldots A_N
B_1 B_2 \ldots B_N

Output

Print the count modulo 998244353.

Constraints

```

### `lcb/arc188_a`

```
For a non-empty string T consisting of A, B, and C, we call it a good string if it can be turned into an empty string by performing the following two types of operations any number of times in any order.

- Operation 1: Choose two identical characters in the string and delete them (cannot be performed if there are not two or more identical characters).
- Operation 2: Choose one A, one B, and one C in the string and delete them (cannot be performed if there are not one or more of each of A, B, and C).

For example, ABACA is a good string because it can be turned into an empty string by performing the operations as follows:

- Choose the 2nd, 4th, and 5th characters and delete them (Operation 2). The string becomes AA.
- Choose the 1st and 2nd characters and delete them (Operation 1). The string becomes an empty string.

You are given a string S of length N consisting of A, B, C, and ?. How many ways are there to replace each ? with A, B, or C to form a string that contains at least K good strings as contiguous substrings? Substrings are counted separately if they are at different positions in the original string, even if they are identical strings.
Find the count modulo 998244353.

Input

The input is given from Standard Input in the following format:
N K
S

Output

Print the answer modulo 998244353.

Constraints


- 1 \leq N \leq 50
- 0 \leq K \leq \frac{N(N+1)}{2}
- N and K are integers.
- |S| = N
- S is a string consisting of A, B, C, and ?.

Sample Input 1

4 2
```

## 2024-11-30  —  3 problem(s)

### `lcb/abc382_d`

```
You are given integers N and M.
Print all integer sequences (A_1, A_2, \ldots, A_N) of length N that satisfy all of the following conditions, in lexicographical order.

- 1 \leq A_i
- A_{i - 1} + 10 \leq A_i for each integer i from 2 through N
- A_N \leq M

What is lexicographical order?
A sequence S = (S_1, S_2, \ldots, S_N) of length N is smaller in lexicographical order than a sequence T = (T_1, T_2, \ldots, T_N) of length N if and only if there exists an integer 1 \leq i \leq N such that both of the following hold:

-  (S_1, S_2, \ldots, S_{i-1}) = (T_1, T_2, \ldots, T_{i-1})
-  S_i is less than T_i (as a number).

Input

The input is given from Standard Input in the following format:
N M

Output

Let X be the number of integer sequences that satisfy the conditions, and print X + 1 lines.
The first line should contain the value of X.
The (i + 1)-th line (1 \leq i \leq X) should contain the i-th smallest integer sequence in lexicographical order, with elements separated by spaces.

Constraints


- 2 \leq N \leq 12
- 10N - 9 \leq M \leq 10N
- All input values are integers.

Sample Input 1

3 23

```

### `lcb/abc382_f`

```
There is a grid with H rows and W columns.
Let (i,j) denote the cell at the i-th row from the top and the j-th column from the left.
There are N horizontal bars numbered from 1 to N placed on the grid.
Bar i consists of L_i blocks of size 1 \times 1 connected horizontally, and its leftmost block is initially at cell (R_i, C_i).
That is, initially, bar i occupies the cells (R_i, C_i), (R_i, C_i + 1), \dots, (R_i, C_i + L_i - 1).
It is guaranteed that there is no cell occupied by two different bars.
The current time is t = 0.
At every time t = 0.5 + n for some non-negative integer n, the following occurs in order of i = 1, 2, \dots, N:

- If bar i is not on the bottom row (the H-th row), and none of the cells directly below the cells occupied by bar i is occupied by any bar, then bar i moves down by one cell. That is, if at that time bar i occupies the cells (r,C_i),(r,C_i+1),\dots,(r,C_i+L_i-1)\ (r < H), and the cell (r + 1, C_i + j) is not occupied by any bar for all j (0 \leq j \leq L_i - 1), then bar i now occupies (r + 1, C_i), (r + 1, C_i + 1), \dots, (r + 1, C_i + L_i - 1).
- Otherwise, nothing happens.

Let (R'_i, C_i), (R'_i, C_i + 1), \dots, (R'_i, C_i + L_i - 1) be the cells occupied by bar i at time t = 10^{100}. Find R'_1, R'_2, \dots, R'_N.

Input

The input is given from Standard Input in the following format:
H W N
R_1 C_1 L_1
R_2 C_2 L_2
\vdots
R_N C_N L_N

Output

Print N lines.
The i-th line (1 \leq i \leq N) should contain R'_i.

Constraints


- 1 \leq H, W \leq 2 \times 10^5
- 1 \leq N \leq 2 \times 10^5
- 1 \leq R_i \leq H
- 1 \leq C_i \leq W
```

### `lcb/abc382_g`

```
Tiles are laid out covering the two-dimensional coordinate plane.
Each tile is a rectangle, and for each integer triple (i, j, k) satisfying 0 \leq k < K, a corresponding tile is placed according to the following rules:

- When i and j have the same parity (both even or both odd), the tile corresponding to (i, j, k) covers the area where iK \leq x \leq (i + 1)K and jK + k \leq y \leq jK + k + 1.
- When i and j have different parity, the tile corresponding to (i, j, k) covers the area where iK + k \leq x \leq iK + k + 1 and jK \leq y \leq (j + 1)K.

Two tiles are adjacent when their edges have a common segment of positive length.
Starting from the tile containing the point (S_x + 0.5, S_y + 0.5), find the minimum number of times you need to move to an adjacent tile to reach the tile containing the point (T_x + 0.5, T_y + 0.5).
There are T test cases; solve each of them.

Input

The input is given from Standard Input in the following format:
T
\text{case}_1
\vdots
\text{case}_T

Each case is given in the following format:
K S_x S_y T_x T_y

Output

Print T lines. The i-th line should contain the answer for the i-th test case.

Constraints


- 1 \leq T \leq 10^4
- 2 \leq K \leq 10^{16}
- -10^{16} \leq S_x, S_y, T_x, T_y \leq 10^{16}
- All input values are integers.

Sample Input 1

```

## 2024-12-07  —  1 problem(s)

### `lcb/abc383_e`

```
You are given a simple connected undirected graph with N vertices and M edges, where vertices are numbered 1 to N and edges are numbered 1 to M. Edge i (1 \leq i \leq M) connects vertices u_i and v_i bidirectionally and has weight w_i.
For a path, define its weight as the maximum weight of an edge in the path.
Define f(x, y) as the minimum possible path weight of a path from vertex x to vertex y.
You are given two sequences of length K: (A_1, A_2, \ldots, A_K) and (B_1, B_2, \ldots, B_K). It is guaranteed that A_i \neq B_j (1 \leq i,j \leq K).
Permute the sequence B freely so that \displaystyle \sum_{i=1}^{K} f(A_i, B_i) is minimized.

Input

The input is given from Standard Input in the following format:
N M K
u_1 v_1 w_1
u_2 v_2 w_2
\vdots
u_M v_M w_M
A_1 A_2 \ldots A_K
B_1 B_2 \ldots B_K

Output

Print the minimum value of \displaystyle \sum_{i=1}^{K} f(A_i, B_i).

Constraints


- 2 \leq N  \leq 2 \times 10^5
- N-1 \leq M  \leq \min(\frac{N \times (N-1)}{2},2 \times 10^5)
- 1 \leq K \leq N
- 1 \leq u_i<v_i \leq N (1 \leq i \leq M) 
- 1 \leq w_i \leq 10^9
- 1 \leq A_i,B_i \leq N (1 \leq i \leq K)
- The given graph is simple and connected.
- All input values are integers.

Sample Input 1

```

## 2024-12-08  —  3 problem(s)

### `lcb/arc189_c`

```
There are N boxes.
For i = 1, 2, \ldots, N, the i-th box contains A_i red balls and B_i blue balls.
You are also given two permutations P = (P_1, P_2, \ldots, P_N) and Q = (Q_1, Q_2, \ldots, Q_N) of (1, 2, \ldots, N).
Takahashi can repeat the following operation any number of times, possibly zero:

- Choose an integer 1 \leq i \leq N, and take all the balls from the i-th box into his hand.
- Put all the red balls in his hand into the P_i-th box.
- Put all the blue balls in his hand into the Q_i-th box.

His goal is to make a state where all boxes other than the X-th box contain no balls by repeating the above operations.
Determine whether it is possible to achieve his goal, and if possible, print the minimum number of operations needed to achieve it.

Input

The input is given from Standard Input in the following format:
N X
A_1 A_2 \ldots A_N
B_1 B_2 \ldots B_N
P_1 P_2 \ldots P_N
Q_1 Q_2 \ldots Q_N

Output

If it is impossible for Takahashi to achieve a state where all boxes other than the X-th box contain no balls, print -1. If it is possible, print the minimum number of operations needed to achieve it.

Constraints


- 2 \leq N \leq 2 \times 10^5
- 0 \leq A_i, B_i \leq 1
- 1 \leq P_i, Q_i \leq N
- P and Q are permutations of (1, 2, \ldots, N).
- 1 \leq X \leq N
- All input values are integers.

```

### `lcb/arc189_d`

```
There are N slimes lined up in a row from left to right.
For i = 1, 2, \ldots, N, the i-th slime from the left has size A_i.
For each K = 1, 2, \ldots, N, solve the following problem.

Takahashi is the K-th slime from the left in the initial state.
Find the maximum size that he can have after performing the following action any number of times, possibly zero:

- Choose a slime adjacent to him that is strictly smaller than him, and absorb it.
As a result, the absorbed slime disappears, and Takahashi's size increases by the size of the absorbed slime.

When a slime disappears due to absorption, the gap is immediately closed, and the slimes that were adjacent to the disappearing slime (if they exist) become adjacent (see the explanation in Sample Input 1).

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \ldots A_N

Output

Print the answers B_K for each K = 1, 2, \ldots, N, separated by spaces, in the following format:
B_1 B_2 \ldots B_N

Constraints


- 2 \leq N \leq 5 \times 10^5
- 1 \leq A_i \leq 10^9
- All input values are integers.

Sample Input 1

6
4 13 2 3 2 6

```

### `lcb/arc189_b`

```
There are N pieces placed on a number line. Initially, all pieces are placed at distinct coordinates.
The initial coordinates of the pieces are X_1, X_2, \ldots, X_N.
Takahashi can repeat the following operation any number of times, possibly zero.

Choose an integer i such that 1 \leq i \leq N-3, and let M be the midpoint between the positions of the i-th and (i+3)-rd pieces in ascending order of coordinate.
Then, move each of the (i+1)-th and (i+2)-th pieces in ascending order of coordinate to positions symmetric to M.
Under the constraints of this problem, it can be proved that all pieces always occupy distinct coordinates, no matter how one repeatedly performs the operation.

His goal is to minimize the sum of the coordinates of the N pieces.
Find the minimum possible sum of the coordinates of the N pieces after repeating the operations.

Input

The input is given from Standard Input in the following format:
N
X_1 X_2 \ldots X_N

Output

Print the minimum possible sum of the coordinates of the N pieces after repeating the operations.

Constraints


- 4 \leq N \leq 2 \times 10^5
- 0 \leq X_1 < X_2 < \cdots < X_N \leq 10^{12}
- All input values are integers.

Sample Input 1

4
1 5 7 10

Sample Output 1

```

## 2024-12-14  —  3 problem(s)

### `lcb/abc384_g`

```
You are given integer sequences A=(A_1,A_2,\ldots,A_N) and B=(B_1,B_2,\ldots,B_N) of length N, and integer sequences X=(X_1,X_2,\ldots,X_K) and Y=(Y_1,Y_2,\ldots,Y_K) of length K.
For each k=1,2,\ldots,K, find \displaystyle \sum_{i=1}^{X_k} \sum_{j=1}^{Y_k} |A_i-B_j|.

Input

The input is given from Standard Input in the following format:
N
A_1 A_2 \ldots A_N
B_1 B_2 \ldots B_N
K
X_1 Y_1
X_2 Y_2
\vdots
X_K Y_K

Output

Print K lines.
The i-th line (1\le i\le K) should contain the answer for k=i.

Constraints


- 1\le N\le 10^5
- 0\le A_i,B_j\le 2\times 10^8
- 1\le K\le 10^4
- 1\le X_k,Y_k\le N
- All input values are integers.

Sample Input 1

2
2 4
3 5
4
```

### `lcb/abc384_e`

```
There is a grid with H horizontal rows and W vertical columns.
Let (i, j) denote the cell at the i-th row (1\leq i\leq H) from the top and j-th column (1\leq j\leq W) from the left.
Initially, there is a slime with strength S _ {i,j} in cell (i,j), and Takahashi is the slime in the cell (P,Q).
Find the maximum possible strength of Takahashi after performing the following action any number of times (possibly zero):

- Among the slimes adjacent to him, choose one whose strength is strictly less than \dfrac{1}{X} times his strength and absorb it.
  As a result, the absorbed slime disappears, and Takahashi's strength increases by the strength of the absorbed slime.

When performing the above action, the gap left by the disappeared slime is immediately filled by Takahashi, and the slimes that were adjacent to the disappeared one (if any) become newly adjacent to Takahashi (refer to the explanation in sample 1).

Input

The input is given in the following format from Standard Input:
H W X 
P Q
S _ {1,1} S _ {1,2} \ldots S _ {1,W}
S _ {2,1} S _ {2,2} \ldots S _ {2,W}
\vdots
S _ {H,1} S _ {H,2} \ldots S _ {H,W}

Output

Print the maximum possible strength of Takahashi after performing the action.

Constraints


- 1\leq H,W\leq500
- 1\leq P\leq H
- 1\leq Q\leq W
- 1\leq X\leq10^9
- 1\leq S _ {i,j}\leq10^{12}
- All input values are integers.

Sample Input 1
```

### `lcb/abc384_f`

```
For a positive integer x, define f(x) as follows: "While x is even, keep dividing it by 2. The final value of x after these divisions is f(x)." For example, f(4)=f(2)=f(1)=1, and f(12)=f(6)=f(3)=3.
Given an integer sequence A=(A_1,A_2,\ldots,A_N) of length N, find \displaystyle \sum_{i=1}^N \sum_{j=i}^N f(A_i+A_j).

Input

The input is given in the following format from Standard Input:
N
A_1 A_2 \ldots A_N

Output

Print the answer.

Constraints


- 1\le N\le 2\times 10^5
- 1\le A_i\le 10^7
- All input values are integers.

Sample Input 1

2
4 8

Sample Output 1

5

f(A_1+A_1)=f(8)=1, f(A_1+A_2)=f(12)=3, f(A_2+A_2)=f(16)=1. Thus, Print 1+3+1=5.

Sample Input 2

3
51 44 63
```

## 2024-12-21  —  3 problem(s)

### `lcb/abc385_f`

```
There are N buildings numbered 1 to N on a number line.
Building i is at coordinate X_i and has height H_i. The size in directions other than height is negligible.
From a point P with coordinate x and height h, building i is considered visible if there exists a point Q on building i such that the line segment PQ does not intersect with any other building.
Find the maximum height at coordinate 0 from which it is not possible to see all buildings. Height must be non-negative; if it is possible to see all buildings at height 0 at coordinate 0, report -1 instead.

Input

The input is given from Standard Input in the following format:
N
X_1 H_1
\vdots
X_N H_N

Output

If it is possible to see all buildings from coordinate 0 and height 0, print -1. Otherwise, print the maximum height at coordinate 0 from which it is not possible to see all buildings. Answers with an absolute or relative error of at most 10^{-9} from the true answer will be considered correct.

Constraints


- 1 \leq N \leq 2 \times 10^5
- 1 \leq X_1 < \dots < X_N \leq 10^9
- 1 \leq H_i \leq 10^9
- All input values are integers.

Sample Input 1

3
3 2
5 4
7 5

Sample Output 1

1.500000000000000000
```

### `lcb/abc385_d`

```
There are N houses at points (X_1,Y_1),\ldots,(X_N,Y_N) on a two-dimensional plane.
Initially, Santa Claus is at point (S_x,S_y). He will act according to the sequence (D_1,C_1),\ldots,(D_M,C_M) as follows:

- For i=1,2,\ldots,M in order, he moves as follows:
- Let (x,y) be the point where he currently is.
- If D_i is U, move in a straight line from (x,y) to (x,y+C_i).
- If D_i is D, move in a straight line from (x,y) to (x,y-C_i).
- If D_i is L, move in a straight line from (x,y) to (x-C_i,y).
- If D_i is R, move in a straight line from (x,y) to (x+C_i,y).





Find the point where he is after completing all actions, and the number of distinct houses he passed through or arrived at during his actions. If the same house is passed multiple times, it is only counted once.

Input

The input is given from Standard Input in the following format:
N M S_x S_y
X_1 Y_1
\vdots
X_N Y_N
D_1 C_1
\vdots
D_M C_M

Output

Let (X,Y) be the point where he is after completing all actions, and C be the number of distinct houses passed through or arrived at. Print X,Y,C in this order separated by spaces.

Constraints


- 1 \leq N \leq 2\times 10^5
```

### `lcb/abc385_e`

```
A "Snowflake Tree" is defined as a tree that can be generated by the following procedure:

- Choose positive integers x,y.
- Prepare one vertex.
- Prepare x more vertices, and connect each of them to the vertex prepared in step 2.
- For each of the x vertices prepared in step 3, attach y leaves to it.

The figure below shows a Snowflake Tree with x=4,y=2. The vertices prepared in steps 2, 3, 4 are shown in red, blue, and green, respectively.

You are given a tree T with N vertices. The vertices are numbered 1 to N, and the i-th edge (i=1,2,\dots,N-1) connects vertices u_i and v_i.
Consider deleting zero or more vertices of T and the edges adjacent to them so that the remaining graph becomes a single Snowflake Tree. Find the minimum number of vertices that must be deleted. Under the constraints of this problem, it is always possible to transform T into a Snowflake Tree.

Input

The input is given from Standard Input in the following format:
N
u_1 v_1
u_2 v_2
\vdots
u_{N-1} v_{N-1}

Output

Print the answer.

Constraints


- 3 \leq N \leq 3 \times 10^5
- 1 \leq u_i < v_i \leq N
- The given graph is a tree.
- All input values are integers.

Sample Input 1

```

## 2024-12-28  —  3 problem(s)

### `lcb/abc386_e`

```
You are given a sequence A of non-negative integers of length N, and an integer K. It is guaranteed that the binomial coefficient \dbinom{N}{K} is at most 10^6.
When choosing K distinct elements from A, find the maximum possible value of the XOR of the K chosen elements.
That is, find \underset{1\leq i_1\lt i_2\lt \ldots\lt i_K\leq N}{\max} A_{i_1}\oplus A_{i_2}\oplus \ldots \oplus A_{i_K}.

About XOR

For non-negative integers A,B, the XOR A \oplus B is defined as follows:


- In the binary representation of A \oplus B, the bit corresponding to 2^k (k \ge 0) is 1 if and only if exactly one of the bits corresponding to 2^k in A and B is 1, and is 0 otherwise.


For example, 3 \oplus 5 = 6 (in binary notation: 011 \oplus 101 = 110).
In general, the XOR of K integers p_1, \dots, p_k is defined as (\cdots((p_1 \oplus p_2) \oplus p_3) \oplus \cdots \oplus p_k). It can be proved that it does not depend on the order of p_1, \dots, p_k.

Input

The input is given from Standard Input in the following format:
N K
A_1 A_2 \ldots A_N

Output

Print the answer.

Constraints


- 1\leq K\leq N\leq 2\times 10^5
- 0\leq A_i<2^{60}
- \dbinom{N}{K}\leq 10^6
- All input values are integers.

Sample Input 1

```

### `lcb/abc386_d`

```
There is an N \times N grid. Takahashi wants to color each cell black or white so that all of the following conditions are satisfied:

- For every row, the following condition holds:

- There exists an integer i\ (0\leq i\leq N) such that the leftmost i cells are colored black, and the rest are colored white.

- For every column, the following condition holds:

- There exists an integer i\ (0\leq i\leq N) such that the topmost i cells are colored black, and the rest are colored white.


Out of these N^2 cells, M of them have already been colored. Among them, the i-th one is at the X_i-th row from the top and the Y_i-th column from the left, and it is colored black if C_i is B and white if C_i is W.
Determine whether he can color the remaining uncolored N^2 - M cells so that all the conditions are satisfied.

Input

The input is given from Standard Input in the following format:
N M
X_1 Y_1 C_1
\vdots
X_M Y_M C_M

Output

If it is possible to satisfy the conditions, print Yes; otherwise, print No.

Constraints


- 1\leq N\leq 10^9
- 1\leq M\leq \min(N^2,2\times 10^5)
- 1\leq X_i,Y_i\leq N
- (X_i,Y_i)\neq (X_j,Y_j)\ (i\neq j)
- C_i is B or W.
- All input numbers are integers.
```

### `lcb/abc386_f`

```
This problem fully contains Problem C (Operate 1), with K \le 20.
You can solve Problem C by submitting a correct solution to this problem for Problem C.
Determine whether it is possible to perform the following operation on string S between 0 and K times, inclusive, to make it identical to string T.

- Choose one of the following three operations and execute it.
- Insert any one character at any position in S (possibly the beginning or end).
- Delete one character from S.
- Choose one character in S and replace it with another character.

Input

The input is given from Standard Input in the following format:
K
S
T

Output

If S can be made identical to T with at most K operations, print Yes; otherwise, print No.

Constraints


- Each of S and T is a string of length between 1 and 500000, inclusive, consisting of lowercase English letters.
- K is an integer satisfying \color{red}{1 \le K \le 20}.

Sample Input 1

3
abc
awtf

Sample Output 1

Yes
```
