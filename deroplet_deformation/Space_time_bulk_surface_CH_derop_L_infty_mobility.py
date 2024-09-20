from ngsolve import *

from Newton_method import Newton_Solve

from netgen.occ import * 

import numpy as np

from netgen.geom2d import SplineGeometry
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


# PDE parameter

Delta=0.02
Sigma=2
Delta_Gamma=0.02
k=1
beta=1
# time parameter
T_Delta=1e-5
T_time=np.arange(0,1+T_Delta,T_Delta)
TLen=len(T_time)
# define the finite element space

Fan_Space=H1(mesh,order=1)
Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer_edges')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()

mass_Fan=np.zeros(TLen-1)
mass_Fan_Gamma=np.zeros(TLen-1)
NewNumb=np.zeros(TLen-1)
Energy_Omega=np.zeros(TLen-1)
Energy_Gamma=np.zeros(TLen-1)
Mu_Energy=np.zeros(TLen-1)
Energy_All=np.zeros(TLen-1)
balance_energy=np.zeros(TLen-1)
mass_Fan_All=np.zeros(TLen-1)
#  write the Newton method

Err=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)
Init_FanL=GridFunction(Fan_Space)
Init_Fan=GridFunction(Fan_Space)

CF1=-1/0.15*((x-0.5)**2/0.2**2+y**2/0.4**2-1)
tanh1=CF(sinh(CF1)/cosh(CF1))
Init_Fan.Set(tanh1)
dS=ds(definedon=mesh.Boundaries('outer_edges'))

def a_variation(fes,Init_Fan):
    a=BilinearForm(fes)
    Fan1=Fan-Init_Fan
    Fan1Trace=Fan.Trace()-Init_Fan.Trace()

    M_Omega=0.1*(1/5*Fan1**4+Init_Fan*Fan1**3+(2*Init_Fan**2-2/3)*Fan1**2+(2*Init_Fan**3-2*Init_Fan)*Fan1+(Init_Fan**4-2*Init_Fan**2))+(0.1+1e-3)
    M_Gamma=0.1*(1/5*Fan1Trace**4+Init_Fan.Trace()*Fan1Trace**3+(2*(Init_Fan.Trace())**2-2/3)*Fan1Trace**2+(2*(Init_Fan.Trace())**3-2*Init_Fan.Trace())*Fan1Trace+((Init_Fan.Trace())**4-2*(Init_Fan.Trace())**2))+(0.1+1e-3)
    
    a+=Fan*theta2*dx+T_Delta*M_Omega*grad(Mu)*grad(theta2)*dx-Init_Fan*theta2*dx\
    +Fan.Trace()*theta3*dS+T_Delta*M_Gamma*grad(Mu_Gamma).Trace()*grad(theta3).Trace()*dS-Init_Fan*theta3*dS\
    -Mu*theta1*dx-Mu_Gamma*theta1.Trace()*dS\
    +Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
    +Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
    +1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
    +k*Delta_Gamma/2*grad(Fan).Trace()*grad(theta1).Trace()*dS\
    +k*Delta_Gamma/2*grad(Init_Fan).Trace()*grad(theta1).Trace()*dS\
    +1/Delta_Gamma*(1/4*(Fan1Trace)**3+Init_Fan.Trace()*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1*dS
    return a

a=a_variation(fes,Init_Fan)
gfu_iter=GridFunction(fes)
gfu_iter.components[0].Set(Init_Fan)

gfu_iterL=GridFunction(fes)
gfu_iterL.components[0].Set(Init_Fan)
vtk = VTKOutput(mesh,coefs=[Init_Fan],names=["sol"],filename="vtk_example1",subdivision=2)
vtk.Do(0)
index1=0
for i in range(TLen-1):
    if (i)%100==0:
       vtk.Do(i)
    index1=index1+1
    print(index1)
    mass_Fan[i]=Integrate(Init_Fan,mesh,order=2)
    mass_Fan_Gamma[i]=Integrate(Init_Fan,mesh,definedon=mesh.Boundaries('outer_edges'),order=2)
    mass_Fan_All[i]=beta*mass_Fan[i]+mass_Fan_Gamma[i]
    Energy_Omega[i]=Integrate(Sigma*Delta*1/2*InnerProduct(grad(Init_Fan),grad(Init_Fan))+Sigma*1/Delta*1/4*(Init_Fan**2-1)**2,mesh,order=2)
    Energy_Gamma[i]=Integrate(k*Delta_Gamma*1/2*InnerProduct(grad(Init_Fan).Trace(),grad(Init_Fan).Trace())+1/Delta_Gamma*1/4*(Init_Fan.Trace()**2-1)**2,mesh,definedon=mesh.Boundaries('outer_edges'),order=2)
    Energy_All[i]=beta*Energy_Omega[i]+Energy_Gamma[i]
    print(Energy_All[i])
    sol,NewNumb[i]=Newton_Solve(a,gfu_iter,tol=1e-10,Max_iter=10)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec

#ErrL2_Omega=sqrt(T_Delta*Err.sum())
#print(ErrL2_Omega)
np.save('mass_Fan_All',mass_Fan_All)
np.save('mass_Fan',mass_Fan)
np.save('mass_Fan_Gamma',mass_Fan_Gamma)
np.save('Energy_Omega',Energy_Omega)
np.save('Energy_Gamma',Energy_Gamma)
np.save('Energy_All',Energy_All)
np.save('NewNumb',NewNumb)