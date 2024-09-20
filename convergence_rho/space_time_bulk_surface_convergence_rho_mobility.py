from ngsolve import *

from Newton_method import Newton_Solve

from netgen.geom2d import SplineGeometry

import numpy as np
# final the versions 
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
T_time=np.arange(0,0.1+T_Delta,T_Delta)
TLen=len(T_time)

K_para=10**(-7)
L_para=1000
alpha=1
beta=1
# define the finite element space

Fan_Space=H1(mesh,order=1)
Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer')))

Psi_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

fesL=Fan_Space*Mu_Space*Mu_Gamma_Space*Psi_Space

(FanL,MuL,Mu_GammaL,PsiL),(theta1L,theta2L,theta3L,theta4L)=fesL.TnT()

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()
Newton_step_numb=np.zeros(TLen-1)
#  write the Newton method

ErrL2_Phi_Omega=np.zeros(TLen-1)
ErrL2_Phi_Gamma=np.zeros(TLen-1)
ErrL2_Psi_Gamma=np.zeros(TLen-1)

ErrH1_Phi_Omega=np.zeros(TLen-1)
ErrH1_Phi_Gamma=np.zeros(TLen-1)
ErrH1_Psi_Gamma=np.zeros(TLen-1)


Init_Fan=GridFunction(Fan_Space)

Init_FanL=GridFunction(Fan_Space)
Init_PsiL=GridFunction(Psi_Space)

Ini_mid=GridFunction(Fan_Space)

Normal=specialcf.normal(2)

Fan_cof=0.1*sin(2*pi*(x**2+y**2))
cof_CF_Fan=CF(Fan_cof)
Init_Fan.Set(cof_CF_Fan)

Init_FanL.Set(cof_CF_Fan)

psi_cof=1/alpha*(Delta*Sigma*K_para*(Fan_cof.Diff(x)*Normal[0]+Fan_cof.Diff(y)*Normal[1])+Fan_cof)
Ini_mid.Set(psi_cof)

for i in range(256):
    Init_PsiL.vec.data[i]=Ini_mid.vec[i]
#
def a_variation_L(fes,Fan,Mu,Mu_Gamma,Psi,theta1,theta2,theta3,theta4,Init_Fan,Init_Psi):
    a_L=BilinearForm(fes)
    Fan1=Fan-Init_Fan
    Psi1=Psi-Init_Psi
    M_Omega=0.1*(1/5*Fan1**4+Init_Fan*Fan1**3+(2*Init_Fan**2-2/3)*Fan1**2+(2*Init_Fan**3-2*Init_Fan)*Fan1+(Init_Fan**4-2*Init_Fan**2))+(0.1+1e-3)
    M_Gamma=0.1*(1/5*Psi1**4+Init_Psi*Psi1**3+(2*Init_Psi**2-2/3)*Psi1**2+(2*Init_Psi**3-2*Init_Psi)*Psi1+(Init_Psi**4-2*Init_Psi**2))+(0.1+1e-3)

    a_L+=Fan*theta2*dx+T_Delta*M_Omega*InnerProduct(grad(Mu),grad(theta2))*dx-Init_Fan*theta2*dx\
     -T_Delta*1/L_para*(beta*Mu_Gamma-Mu)*theta2*ds\
     +Psi*theta3*ds+T_Delta*M_Gamma*InnerProduct(grad(Mu_Gamma).Trace(),grad(theta3).Trace())*ds-Init_Psi*theta3*ds\
     +T_Delta*beta*1/L_para*(beta*Mu_Gamma-Mu)*theta3*ds\
     -Mu*theta1*dx-Mu_Gamma*theta4*ds\
     +Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
     +Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
     +1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
     +k*Delta_Gamma/2*grad(Psi).Trace()*grad(theta4).Trace()*ds\
     +k*Delta_Gamma/2*grad(Init_Psi).Trace()*grad(theta4).Trace()*ds\
     +1/Delta_Gamma*(1/4*(Psi1)**3+Init_Psi*(Psi1)**2+(3/2*(Init_Psi)**2-1/2)*Psi1+(Init_Psi)**3-Init_Psi)*theta4*ds\
     +1/K_para*(alpha*Psi-Fan.Trace())*(alpha*theta4-theta1.Trace())*ds
    return a_L


def a_variation(fes,Fan,Mu,Mu_Gamma,theta1,theta2,theta3,Init_Fan):
   a=BilinearForm(fes)
   Fan1=Fan-Init_Fan

   Fan1Trace=Fan.Trace()-Init_Fan.Trace()
   
   M_Omega=0.1*(1/5*Fan1**4+Init_Fan*Fan1**3+(2*Init_Fan**2-2/3)*Fan1**2+(2*Init_Fan**3-2*Init_Fan)*Fan1+(Init_Fan**4-2*Init_Fan**2))+(0.1+1e-3)
   M_Gamma=0.1*(1/5*Fan1Trace**4+(Init_Fan.Trace())*Fan1Trace**3+(2*(Init_Fan.Trace())**2-2/3)*Fan1Trace**2+(2*(Init_Fan.Trace())**3-2*(Init_Fan.Trace()))*Fan1Trace+((Init_Fan.Trace())**4-2*(Init_Fan.Trace())**2))+(0.1+1e-3)

   a+=Fan*theta2*dx+T_Delta*M_Omega*InnerProduct(grad(Mu),grad(theta2))*dx-Init_Fan*theta2*dx\
   -T_Delta*1/L_para*(beta*Mu_Gamma-Mu)*theta2*ds\
   +Fan.Trace()*theta3*ds+T_Delta*M_Gamma*InnerProduct(grad(Mu_Gamma).Trace(),grad(theta3).Trace())*ds-Init_Fan*theta3*ds\
   +T_Delta*beta*1/L_para*(beta*Mu_Gamma-Mu)*theta3*ds\
   -Mu*theta1*dx-Mu_Gamma*theta1.Trace()*ds\
   +Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
   +Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
   +1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
   +k*Delta_Gamma/2*grad(Fan).Trace()*grad(theta1).Trace()*ds\
   +k*Delta_Gamma/2*grad(Init_Fan).Trace()*grad(theta1).Trace()*ds\
   +1/Delta_Gamma*(1/4*(Fan1Trace)**3+(Init_Fan.Trace())*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1.Trace()*ds
   return a


a=a_variation(fes,Fan,Mu,Mu_Gamma,theta1,theta2,theta3,Init_Fan)
a_L=a_variation_L(fesL,FanL,MuL,Mu_GammaL,PsiL,theta1L,theta2L,theta3L,theta4L,Init_FanL,Init_PsiL)
#a_L=a_variation_L(fes,Fan,Mu,Mu_Gamma,theta1,theta2,theta3,Init_FanL)

gfu_iter=GridFunction(fes)
gfu_iter.components[0].Set(Init_Fan)

gfu_iterL=GridFunction(fesL)
for i in range(256):
    gfu_iterL.components[3].vec.data[i]=Ini_mid.vec[i]

index1=0
for i in range(TLen-1):
    index1=index1+1
    print(index1)
    sol,Newton_step_numb[i]=Newton_Solve(a,gfu_iter,tol=1e-12,Max_iter=10)
    Init_Fan.vec.data=sol.components[0].vec
    
    solL,Newton_step_numb[i]=Newton_Solve(a_L,gfu_iterL,tol=1e-12,Max_iter=10)

    Init_FanL.vec.data=solL.components[0].vec
    Init_PsiL.vec.data=solL.components[3].vec
    
    ErrL2_Phi_Omega[i]=Integrate((Init_Fan-Init_FanL)**2,mesh,order=2)
    ErrL2_Phi_Gamma[i]=Integrate((Init_Fan-Init_FanL)**2,mesh,definedon=mesh.Boundaries('outer'),order=2)
    ErrL2_Psi_Gamma[i]=Integrate((Init_Fan-Init_PsiL)**2,mesh,definedon=mesh.Boundaries('outer'),order=2)

    ErrH1_Phi_Omega[i]=Integrate((Init_Fan-Init_FanL)**2+InnerProduct(grad(Init_Fan)-grad(Init_FanL),grad(Init_Fan)-grad(Init_FanL)),mesh,order=2)
    ErrH1_Phi_Gamma[i]=Integrate((Init_Fan-Init_FanL)**2+InnerProduct(grad(Init_Fan).Trace()-grad(Init_FanL).Trace(),grad(Init_Fan).Trace()-grad(Init_FanL).Trace()),mesh,definedon=mesh.Boundaries('outer'),order=2)
    ErrH1_Psi_Gamma[i]=Integrate((Init_Fan-Init_PsiL)**2+InnerProduct(grad(Init_Fan).Trace()-grad(Init_PsiL).Trace(),grad(Init_Fan).Trace()-grad(Init_PsiL).Trace()),mesh,definedon=mesh.Boundaries('outer'),order=2)

AErrL2_Phi_Omega=sqrt(T_Delta*ErrL2_Phi_Omega.sum())
AErrL2_Phi_Gamma=sqrt(T_Delta*ErrL2_Phi_Gamma.sum())
AErrL2_Psi_Gamma=sqrt(T_Delta*ErrL2_Psi_Gamma.sum())

AErrH1_Phi_Omega=sqrt(T_Delta*ErrH1_Phi_Omega.sum())
AErrH1_Phi_Gamma=sqrt(T_Delta*ErrH1_Phi_Gamma.sum())
AErrH1_Psi_Gamma=sqrt(T_Delta*ErrH1_Psi_Gamma.sum())

print([AErrL2_Phi_Omega,AErrL2_Phi_Gamma,AErrL2_Psi_Gamma,AErrH1_Phi_Omega,AErrH1_Phi_Gamma,AErrH1_Psi_Gamma])