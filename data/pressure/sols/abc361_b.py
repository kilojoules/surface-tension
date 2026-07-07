import sys
v = sys.stdin.read().split()
a,b,c,d,e,f = v[0:6]
g,h,i,j,k,l = v[6:12]
a,b,c,d,e,f = int(a),int(b),int(c),int(d),int(e),int(f)
g,h,i,j,k,l = int(g),int(h),int(i),int(j),int(k),int(l)
x = min(d,j) - max(a,g)
y = min(e,k) - max(b,h)
z = min(f,l) - max(c,i)
print("Yes" if x > 0 and y > 0 and z > 0 else "No")
