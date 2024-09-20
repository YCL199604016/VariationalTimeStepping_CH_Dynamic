from ngsolve import *

from Newton_method import Newton_Solve

from netgen.geom2d import SplineGeometry

import numpy as np
from netgen.occ import * 

#  inital value is bu ke dao , zhe shi bu xing de
########################### mesh###############################
outer = Rectangle(1, 1).Face()
outer.edges.name="outer_edges"

mid=Rectangle(1,0.6).Face()
mid.edges.Max(X).name = "outer_edges"
mid.edges.Min(X).name = "outer_edges"
mid.edges.Min(Y).name = "outer_edges"
mid.edges.Max(Y).name = "interface"
mid.faces.name='mid'
mid.maxh=1/2**6
outer1=outer-mid
outer1.faces.name='outer'

geo = Glue([outer1,mid])
geo1=OCCGeometry(geo, dim=2)
mesh1=geo1.GenerateMesh(maxh=1/2**4)
mesh = Mesh(mesh1)
################################# mesh ##############################
# PDE parameter
Diff_CF=0.02
Delta=0.02
Sigma=2
Diff_Gamma_CF=0.02
Delta_Gamma=0.02
k=1
L_para=10
beta=1
#note the constrain of time step to assure exist of solution
# time parameter
T_Delta=1e-5
T_time=np.arange(0,2+T_Delta,T_Delta)
TLen=len(T_time)

# define the finite element space

Fan_Space=H1(mesh,order=1)
Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer_edges')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()

#  write the Newton method
mass_Fan=np.zeros(TLen-1)
mass_Fan_Gamma=np.zeros(TLen-1)
NewNumb=np.zeros(TLen-1)
Energy_Omega=np.zeros(TLen-1)
Energy_Gamma=np.zeros(TLen-1)
Mu_Energy=np.zeros(TLen-1)
Energy_All=np.zeros(TLen-1)
balance_energy=np.zeros(TLen-1)
mass_Fan_All=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)
CF1=-1/0.15*((x-0.5)**2/0.2**2+y**2/0.4**2-1)
tanh1=CF(sinh(CF1)/cosh(CF1))
Init_Fan.Set(tanh1)
a=BilinearForm(fes)

Fan1=Fan-Init_Fan

Fan1Trace=Fan.Trace()-Init_Fan.Trace()
dS=ds(definedon='outer_edges')

a+=Fan*theta2*dx+T_Delta*Diff_CF*grad(Mu)*grad(theta2)*dx-Init_Fan*theta2*dx\
-T_Delta*1/L_para*Diff_CF*(beta*Mu_Gamma-Mu)*theta2*dS\
+Fan.Trace()*theta3*dS+T_Delta*Diff_Gamma_CF*grad(Mu_Gamma).Trace()*grad(theta3).Trace()*dS-Init_Fan.Trace()*theta3*dS\
+T_Delta*beta*Diff_Gamma_CF*1/L_para*(beta*Mu_Gamma-Mu)*theta3*dS\
-Mu*theta1*dx-Mu_Gamma*theta1.Trace()*dS\
+Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
+Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
+1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
+k*Delta_Gamma/2*grad(Fan).Trace()*grad(theta1).Trace()*dS\
+k*Delta_Gamma/2*grad(Init_Fan).Trace()*grad(theta1).Trace()*dS\
+1/Delta_Gamma*(1/4*(Fan1Trace)**3+Init_Fan.Trace()*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1*dS

gfu_iter=GridFunction(fes)

gfu_iter.components[0].Set(Init_Fan)
vtk = VTKOutput(mesh,coefs=[Init_Fan],names=["sol"],filename="vtk_example1",subdivision=2)
index1=0
vtk.Do(0)
for i in range(TLen-1):
    if (i)%100==0:
       vtk.Do(i)
    sol,NewNumb[i]=Newton_Solve(a,gfu_iter,tol=1e-14,Max_iter=100)
    mass_Fan[i]=Integrate(Init_Fan,mesh,order=2)
    mass_Fan_Gamma[i]=Integrate(Init_Fan,mesh,definedon=mesh.Boundaries('outer_edges'),order=2)
    mass_Fan_All[i]=beta*mass_Fan[i]+mass_Fan_Gamma[i]
    Energy_Omega[i]=Integrate(Sigma*Delta*1/2*InnerProduct(grad(Init_Fan),grad(Init_Fan))+Sigma*1/Delta*1/4*(Init_Fan**2-1)**2,mesh,order=2)
    Energy_Gamma[i]=Integrate(k*Delta_Gamma*1/2*InnerProduct(grad(Init_Fan).Trace(),grad(Init_Fan).Trace())+1/Delta_Gamma*1/4*(Init_Fan.Trace()**2-1)**2,mesh,definedon=mesh.Boundaries('outer_edges'),order=2)
    Energy_All[i]=beta*Energy_Omega[i]+Energy_Gamma[i]
    print(Energy_All[i])
    index1+=1
    print(index1)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec

np.save('mass_Fan_All',mass_Fan_All)
np.save('mass_Fan',mass_Fan)
np.save('mass_Fan_Gamma',mass_Fan_Gamma)
np.save('Energy_Omega',Energy_Omega)
np.save('Energy_Gamma',Energy_Gamma)
np.save('Energy_All',Energy_All)
np.save('NewNumb',NewNumb)
