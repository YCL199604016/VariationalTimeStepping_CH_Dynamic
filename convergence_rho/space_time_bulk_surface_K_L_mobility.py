from ngsolve import *

from Newton_method import Newton_Solve

from netgen.geom2d import SplineGeometry

import numpy as np

geo = SplineGeometry()
geo.AddRectangle( (0, 0), (1, 1), bc = "outer")
mesh = Mesh( geo.GenerateMesh(maxh=1/2**6))


# PDE parameter

Delta=0.02
Sigma=2
Delta_Gamma=0.02
k=1

# time parameter
T_Delta=1e-5
T_time=np.arange(0,1+T_Delta,T_Delta)
TLen=len(T_time)

K_para=10
L_para=1000
alpha=1
beta=1
# define the finite element space

Fan_Space=H1(mesh,order=1)

Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer')))

Psi_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space*Psi_Space

(Fan,Mu,Mu_Gamma,Psi),(theta1,theta2,theta3,theta4)=fes.TnT()

NewNumb=np.zeros(TLen-1)
#  write the Newton method
mass_Fan=np.zeros(TLen-1)
mass_Psi=np.zeros(TLen-1)
mass_All=np.zeros(TLen-1)

Energy_Omega=np.zeros(TLen-1)
Energy_Gamma=np.zeros(TLen-1)
Energy_Bound=np.zeros(TLen-1)

Mu_Energy=np.zeros(TLen-1)
Energy_All=np.zeros(TLen-1)
balance_energy=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)
Init_Psi=GridFunction(Psi_Space)
Ini_mid=GridFunction(Fan_Space)
Normal=specialcf.normal(2)

Fan_cof=0.1*sin(2*pi*(x**2+y**2))
cof_CF_Fan=CF(Fan_cof)
Init_Fan.Set(cof_CF_Fan)

psi_cof=1/alpha*(Delta*Sigma*K_para*(Fan_cof.Diff(x)*Normal[0]+Fan_cof.Diff(y)*Normal[1])+Fan_cof)
Ini_mid.Set(psi_cof)

for i in range(256):
    Init_Psi.vec.data[i]=Ini_mid.vec[i]

a=BilinearForm(fes)
Fan1=Fan-Init_Fan

Psi1=Psi-Init_Psi

M_Omega=0.1*(1/5*Fan1**4+Init_Fan*Fan1**3+(2*Init_Fan**2-2/3)*Fan1**2+(2*Init_Fan**3-2*Init_Fan)*Fan1+(Init_Fan**4-2*Init_Fan**2))+(0.1+1e-3)

M_Gamma=0.1*(1/5*Psi1**4+Init_Psi*Psi1**3+(2*Init_Psi**2-2/3)*Psi1**2+(2*Init_Psi**3-2*Init_Psi)*Psi1+(Init_Psi**4-2*Init_Psi**2))+(0.1+1e-3)

a+=Fan*theta2*dx+T_Delta*M_Omega*InnerProduct(grad(Mu),grad(theta2))*dx-Init_Fan*theta2*dx\
-T_Delta*1/L_para*(beta*Mu_Gamma-Mu)*theta2*ds\
+Psi*theta3*ds+T_Delta*M_Gamma*grad(Mu_Gamma).Trace()*grad(theta3).Trace()*ds-Init_Psi*theta3*ds\
+T_Delta*beta*1/L_para*(beta*Mu_Gamma-Mu)*theta3*ds\
-Mu*theta1*dx-Mu_Gamma*theta4*ds\
+Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
+Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
+1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
+k*Delta_Gamma/2*grad(Psi).Trace()*grad(theta4).Trace()*ds\
+k*Delta_Gamma/2*grad(Init_Psi).Trace()*grad(theta4).Trace()*ds\
+1/Delta_Gamma*(1/4*(Psi1)**3+Init_Psi*(Psi1)**2+(3/2*(Init_Psi)**2-1/2)*Psi1+(Init_Psi)**3-Init_Psi)*theta4*ds\
+1/K_para*(alpha*Psi-Fan.Trace())*(alpha*theta4-theta1.Trace())*ds

gfu_iter=GridFunction(fes)
gfu_iter.components[0].Set(Init_Fan)

for i in range(256):
    gfu_iter.components[3].vec.data[i]=Ini_mid.vec[i]

vtk = VTKOutput(mesh,coefs=[Init_Fan],names=["sol"],filename="vtk_example1",subdivision=2)

index1=0
vtk.Do(0)
for i in range(TLen-1):
    if (i)%100==0:
       vtk.Do(i)
    mass_Fan[i]=Integrate(Init_Fan,mesh,order=2)
    mass_Psi[i]=Integrate(Init_Psi,mesh,definedon=mesh.Boundaries('outer'),order=2)
    mass_All[i]=beta*mass_Fan[i]+mass_Psi[i]
    Energy_Omega[i]=Integrate(Sigma*Delta*1/2*InnerProduct(grad(Init_Fan),grad(Init_Fan))+Sigma*1/Delta*1/4*(Init_Fan**2-1)**2,mesh,order=2)
    Energy_Gamma[i]=Integrate(k*Delta_Gamma*1/2*InnerProduct(grad(Init_Psi).Trace(),grad(Init_Psi).Trace())+1/Delta_Gamma*1/4*(Init_Psi**2-1)**2,mesh,definedon=mesh.Boundaries('outer'),order=2)
    Energy_Bound[i]=1/(2*K_para)*Integrate((alpha*Init_Psi-Init_Fan.Trace())**2,mesh,definedon=mesh.Boundaries('outer'),order=2)
    Energy_All[i]=Energy_Omega[i]+Energy_Gamma[i]+Energy_Bound[i]
    index1+=1
    print(index1)
    sol,NewNumb[i]=Newton_Solve(a,gfu_iter,tol=1e-10,Max_iter=10)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec
    Init_Psi.vec.data=sol.components[3].vec
    print(Energy_All[i])


np.save('mass_Fan',mass_Fan)
np.save('mass_Psi',mass_Psi)
np.save('Energy_Omega',Energy_Omega)
np.save('Energy_Gamma',Energy_Gamma)
np.save('Energy_All',Energy_All)
np.save('balance_energy',balance_energy)
np.save('NewNumb',NewNumb)
np.save('mass_All',mass_All)