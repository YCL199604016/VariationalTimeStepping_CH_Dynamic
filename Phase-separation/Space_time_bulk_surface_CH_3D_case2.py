from ngsolve import *

from Newton_method import Newton_Solve

from netgen.geom2d import SplineGeometry

import numpy as np
from netgen.occ import *
sphere = Sphere((0,0,0),1/2)
sphere.faces.name='outer'
geo = OCCGeometry(sphere)
mesh = Mesh(geo.GenerateMesh(maxh=1/2**5))
mesh.Curve(2)

# PDE parameter

Diff_CF=0.01
Delta=0.01
Sigma=2
Diff_Gamma_CF=0.02
Delta_Gamma=0.01
k=1

# time parameter
T_Delta=1e-4
T_time=np.arange(0,0.2+T_Delta,T_Delta)
TLen=len(T_time)

# define the finite element space

Fan_Space=H1(mesh,order=2)
Mu_Space=H1(mesh,order=2)

Mu_Gamma_Space=Compress(H1(mesh,order=2,definedon=mesh.Boundaries('outer')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()

Newton_step_numb=np.zeros(TLen-1)
#  write the Newton method
mass_Fan=np.zeros(TLen-1)
mass_Fan_Gamma=np.zeros(TLen-1)

Energy_Omega=np.zeros(TLen-1)
Energy_Gamma=np.zeros(TLen-1)
Mu_Energy=np.zeros(TLen-1)
Energy_All=np.zeros(TLen-1)
balance_energy=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)

#CF1=0.1*sin(2*pi*x)*sin(2*pi*y)
Init_Fan.Set(0.2*sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*z))

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
+1/Delta_Gamma*(1/4*(Fan1Trace)**3+Init_Fan.Trace()*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1*ds

gfu_iter=GridFunction(fes)

gfu_iter.components[0].Set(Init_Fan)

vtk = VTKOutput(mesh,coefs=[Init_Fan],names=["sol"],filename="vtk_example1",subdivision=2)
index1=0
vtk.Do(1)
for i in range(TLen-1):
    if (i)%2==0:
       vtk.Do(i)
    sol,Newton_step_numb[i]=Newton_Solve(a,gfu_iter,tol=1e-10,Max_iter=10)
    mass_Fan[i]=Integrate(Init_Fan,mesh,order=2)
    mass_Fan_Gamma[i]=Integrate(Init_Fan,mesh,definedon=mesh.Boundaries('outer'),order=2)
    
    Energy_Omega[i]=Integrate(Sigma*Delta*1/2*InnerProduct(grad(Init_Fan),grad(Init_Fan))+Sigma*1/Delta*1/4*(Init_Fan**2-1)**2,mesh,order=2)
    Energy_Gamma[i]=Integrate(k*Delta_Gamma*1/2*InnerProduct(grad(Init_Fan).Trace(),grad(Init_Fan).Trace())+1/Delta_Gamma*1/4*(Init_Fan.Trace()**2-1)**2,mesh,definedon=mesh.Boundaries('outer'),order=2)
    Energy_All[i]=Energy_Omega[i]+Energy_Gamma[i]

    Mu_Omega_int=Integrate(InnerProduct(grad(sol.components[1]),grad(sol.components[1])),mesh,order=2)

    Mu_Gamma_int=Integrate(InnerProduct(grad(sol.components[2]).Trace(),grad(sol.components[2]).Trace()),mesh,definedon=mesh.Boundaries('outer'),order=2)
    partial_energy=T_Delta*(Diff_CF*Mu_Omega_int+Diff_Gamma_CF*Mu_Gamma_int)
    if i==0:
       Mu_Energy[i]=partial_energy
    else:
       Mu_Energy[i]=partial_energy+Mu_Energy[i-1]
       balance_energy[i-1]=Mu_Energy[i-1]+Energy_All[i]
       #print(balance_energy[i-1])
    index1+=1
    print(index1)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec
    print(Energy_All[i])

np.save('mass_Fan',mass_Fan)
np.save('mass_Fan_Gamma',mass_Fan_Gamma)
np.save('Energy_Omega',Energy_Omega)
np.save('Energy_Gamma',Energy_Gamma)
np.save('Energy_All',Energy_All)
np.save('balance_energy',balance_energy)
np.save('Newton_step_numb',Newton_step_numb)