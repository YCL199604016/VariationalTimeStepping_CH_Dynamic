from ngsolve import *

from Newton_method import Newton_Solve

from netgen.geom2d import SplineGeometry

import numpy as np
from bulk_Surface_exact_sol import sol_token
geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1, 1), bc = "outer")
sizeh=7
h=1/2**sizeh
mesh = Mesh( geo.GenerateMesh(maxh=h))

Normal=specialcf.normal(mesh.dim)
t=Parameter(0)
# PDE parameter
# time parameter
T_Delta=(h)**(1)
T_time=np.arange(0,1+T_Delta,T_Delta)
TLen=len(T_time)
#
Diff_CF,Delta,Sigma,Diff_Gamma_CF,Delta_Gamma,k,Phi_0,Phi_ex,F_Bulk_right,G_Gamma_right=sol_token(t,T_Delta,Normal)
# define the finite element space

Fan_Space=H1(mesh,order=1)
Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()

#  write the Newton method
mass_Fan=np.zeros(TLen-1)
mass_Fan_Gamma=np.zeros(TLen-1)
Err_Gamma_L2=np.zeros(TLen-1)
Err_Omega_L2=np.zeros(TLen-1)
H1_Err_Omega=np.zeros(TLen-1)
H1_Err_Gamma=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)
Init_Fan.Set(Phi_0)

a=BilinearForm(fes)
Fan1=Fan-Init_Fan
Fan1Trace=Fan.Trace()-Init_Fan.Trace()

a+=Fan*theta2*dx+T_Delta*Diff_CF*grad(Mu)*grad(theta2)*dx-Init_Fan*theta2*dx\
+Fan.Trace()*theta3*ds+T_Delta*Diff_Gamma_CF*grad(Mu_Gamma).Trace()*grad(theta3).Trace()*ds-Init_Fan*theta3*ds\
-Mu*theta1*dx-Mu_Gamma*theta1.Trace()*ds\
+Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
+Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
+1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
+k*Delta_Gamma/2*grad(Fan).Trace()*grad(theta1).Trace()*ds\
+k*Delta_Gamma/2*grad(Init_Fan).Trace()*grad(theta1).Trace()*ds\
+1/Delta_Gamma*(1/4*(Fan1Trace)**3+Init_Fan.Trace()*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1*ds\
+F_Bulk_right*theta1*dx+G_Gamma_right*theta1.Trace()*ds
# hou mian bu shi er jie jingsi
gfu_iter=GridFunction(fes)
gfu_iter.components[0].Set(Init_Fan)

gfuPhi_ex=GridFunction(Fan_Space)
vtk = VTKOutput(mesh,coefs=[Init_Fan],names=["sol"],filename="vtk_example1",subdivision=2)
index1=0
vtk.Do(1)

for i in range(TLen-1):
    t.Set(T_time[i+1])
    gfuPhi_ex.Set(Phi_ex)
    if (i)%10==0:
       vtk.Do(i)
    sol,H=Newton_Solve(a,gfu_iter,tol=1e-14,Max_iter=10)
    mass_Fan[i]=Integrate(Init_Fan,mesh,order=2)
    mass_Fan_Gamma[i]=Integrate(Init_Fan,mesh,definedon=mesh.Boundaries('outer'),order=2)
    index1+=1
    print(index1)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec
    Err_Omega_L2[i]=Integrate((Init_Fan-Phi_ex)**2,mesh)
    Err_Gamma_L2[i]=Integrate((Init_Fan-Phi_ex)**2,mesh,definedon=mesh.Boundaries('outer'),order=2)
    #H1_Err_Omega[i]=Integrate((Init_Fan-gfuPhi_ex)**2+InnerProduct(grad(Init_Fan)-grad(gfuPhi_ex),grad(Init_Fan)-grad(gfuPhi_ex)),mesh,order=2)
    #H1_Err_Gamma[i]=Integrate((Init_Fan-gfuPhi_ex)**2+InnerProduct(grad(Init_Fan).Trace()-grad(gfuPhi_ex).Trace(),grad(Init_Fan).Trace()-grad(gfuPhi_ex).Trace()),mesh,definedon=mesh.Boundaries('outer'),order=2)
l2_Err_Omega=sqrt(T_Delta*Err_Omega_L2.sum())
l2_Err_Gamma=sqrt(T_Delta*Err_Gamma_L2.sum())
#Err_Omega_H1=sqrt(T_Delta*H1_Err_Omega.sum())
#Err_Gamma_H1=sqrt(T_Delta*H1_Err_Gamma.sum())

print([l2_Err_Omega,l2_Err_Gamma])
#print([Err_Omega_H1,Err_Gamma_H1])
