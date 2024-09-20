from ngsolve import *
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
import numpy as np

def Newton_Solve(a,gfu_iter,tol=1e-10,Max_iter=100):
    res=gfu_iter.vec.CreateVector()
    du=gfu_iter.vec.CreateVector()
    for i in range(Max_iter):
       res=a.Apply(gfu_iter.vec)
       a.AssembleLinearization(gfu_iter.vec)
       #aMat=sp.csc_matrix(a.mat.CSR())
       #print(aMat.shape)
       du=-a.mat.Inverse(inverse='pardiso')*res
       gfu_iter.vec.data+=du
       Stop_Condition=sqrt(abs(InnerProduct(du,res)))
       #print('solve_linear_equation',Stop_Condition)
       if Stop_Condition<=tol:
             return gfu_iter,i+1