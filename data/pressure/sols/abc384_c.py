import sys
import numpy as np

s = np.array(sys.stdin.read().split(), dtype=np.int64)
names = np.array(['A','AB','ABC','ABCD','ABCDE','ABCE','ABD','ABDE','ABE','AC','ACD','ACDE','ACE','AD','ADE','AE','B','BC','BCD','BCDE','BCE','BD','BDE','BE','C','CD','CDE','CE','D','DE','E'])
M = np.array([[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,1,1,1,0],[1,1,1,1,1],[1,1,1,0,1],[1,1,0,1,0],[1,1,0,1,1],[1,1,0,0,1],[1,0,1,0,0],[1,0,1,1,0],[1,0,1,1,1],[1,0,1,0,1],[1,0,0,1,0],[1,0,0,1,1],[1,0,0,0,1],[0,1,0,0,0],[0,1,1,0,0],[0,1,1,1,0],[0,1,1,1,1],[0,1,1,0,1],[0,1,0,1,0],[0,1,0,1,1],[0,1,0,0,1],[0,0,1,0,0],[0,0,1,1,0],[0,0,1,1,1],[0,0,1,0,1],[0,0,0,1,0],[0,0,0,1,1],[0,0,0,0,1]], dtype=np.int64)
scores = M @ s
order = np.argsort(-scores, kind='stable')
sys.stdout.write('\n'.join(names[order]) + '\n')
