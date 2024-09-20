from ngsolve import *

from Newton_method import Newton_Solve

from netgen.occ import * 

import numpy as np

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

# time parameter
T_Delta=1e-5
T_time=np.arange(0,0.1+T_Delta,T_Delta)
TLen=len(T_time)
L_para=1e+6

beta=1
# define the finite element space

Fan_Space=H1(mesh,order=1)
Mu_Space=H1(mesh,order=1)

Mu_Gamma_Space=Compress(H1(mesh,order=1,definedon=mesh.Boundaries('outer_edges')))

fes=Fan_Space*Mu_Space*Mu_Gamma_Space

(Fan,Mu,Mu_Gamma),(theta1,theta2,theta3)=fes.TnT()

Newton_step_numb=np.zeros(TLen-1)
#  write the Newton method

ErrL2Omega=np.zeros(TLen-1)
ErrL2Gamma=np.zeros(TLen-1)
ErrH1Omega=np.zeros(TLen-1)
ErrH1Gamma=np.zeros(TLen-1)

Init_Fan=GridFunction(Fan_Space)

Init_FanL=GridFunction(Fan_Space)

CF1=-1/0.15*((x-0.5)**2/0.2**2+y**2/0.4**2-1)
tanh1=CF(sinh(CF1)/cosh(CF1))
Init_Fan.Set(tanh1)
Init_FanL.Set(tanh1)
dS=ds(definedon=mesh.Boundaries('outer_edges'))

def a_variation_L(fes,Init_Fan):
    a_L=BilinearForm(fes)
    Fan1=Fan-Init_Fan
    Fan1Trace=Fan.Trace()-Init_Fan.Trace()

    M_Omega=0.1*(1/5*Fan1**4+Init_Fan*Fan1**3+(2*Init_Fan**2-2/3)*Fan1**2+(2*Init_Fan**3-2*Init_Fan)*Fan1+(Init_Fan**4-2*Init_Fan**2))+(0.1+1e-3)

    M_Gamma=0.1*(1/5*Fan1Trace**4+Init_Fan.Trace()*Fan1Trace**3+(2*(Init_Fan.Trace())**2-2/3)*Fan1Trace**2+(2*(Init_Fan.Trace())**3-2*Init_Fan.Trace())*Fan1Trace+((Init_Fan.Trace())**4-2*(Init_Fan.Trace())**2))+(0.1+1e-3)

    a_L+=Fan*theta2*dx+T_Delta*M_Omega*InnerProduct(grad(Mu),grad(theta2))*dx-Init_Fan*theta2*dx\
     -T_Delta*1/L_para*(beta*Mu_Gamma-Mu)*theta2*dS\
     +Fan.Trace()*theta3*dS+T_Delta*M_Gamma*grad(Mu_Gamma).Trace()*grad(theta3).Trace()*dS-Init_Fan*theta3*dS\
     +T_Delta*beta*1/L_para*(beta*Mu_Gamma-Mu)*theta3*dS\
     -Mu*theta1*dx-Mu_Gamma*theta1.Trace()*dS\
     +Delta/2*Sigma*grad(Fan)*grad(theta1)*dx\
     +Delta/2*Sigma*grad(Init_Fan)*grad(theta1)*dx\
     +1/Delta*Sigma*(1/4*Fan1**3+Init_Fan*Fan1**2+(3/2*Init_Fan**2-1/2)*Fan1+Init_Fan**3-Init_Fan)*theta1*dx\
     +k*Delta_Gamma/2*grad(Fan).Trace()*grad(theta1).Trace()*dS\
     +k*Delta_Gamma/2*grad(Init_Fan).Trace()*grad(theta1).Trace()*dS\
     +1/Delta_Gamma*(1/4*(Fan1Trace)**3+Init_Fan.Trace()*(Fan1Trace)**2+(3/2*(Init_Fan.Trace())**2-1/2)*Fan1Trace+(Init_Fan.Trace())**3-Init_Fan.Trace())*theta1*dS
    return a_L

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
a_L=a_variation_L(fes,Init_FanL)
gfu_iter=GridFunction(fes)
gfu_iter.components[0].Set(Init_Fan)

gfu_iterL=GridFunction(fes)
gfu_iterL.components[0].Set(Init_Fan)

index1=0
for i in range(TLen-1):
    index1=index1+1
    print(index1)
    sol,Newton_step_numb[i]=Newton_Solve(a,gfu_iter,tol=1e-10,Max_iter=10)
    gfu_iter.components[0].vec.data=sol.components[0].vec
    Init_Fan.vec.data=sol.components[0].vec

    solL,Newton_step_numb[i]=Newton_Solve(a_L,gfu_iterL,tol=1e-10,Max_iter=10)
    gfu_iterL.components[0].vec.data=solL.components[0].vec

    Init_FanL.vec.data=solL.components[0].vec
    
    ErrL2Omega[i]=Integrate((Init_Fan-Init_FanL)**2,mesh,order=2)
    ErrL2Gamma[i]=Integrate((Init_Fan-Init_FanL)**2,mesh,definedon=mesh.Boundaries('outer_edges'),order=2)
    ErrH1Omega[i]=Integrate((Init_Fan-Init_FanL)**2+InnerProduct(grad(Init_Fan)-grad(Init_FanL),grad(Init_Fan)-grad(Init_FanL)),mesh,order=2)
    ErrH1Gamma[i]=Integrate((Init_Fan-Init_FanL)**2+InnerProduct(grad(Init_Fan).Trace()-grad(Init_FanL).Trace(),grad(Init_Fan).Trace()-grad(Init_FanL).Trace()),mesh,definedon=mesh.Boundaries('outer_edges'),order=2)

ErrL2_all_Omega=sqrt(T_Delta*ErrL2Omega.sum())
ErrL2_all_Gamma=sqrt(T_Delta*ErrL2Gamma.sum())
ErrH1_all_Omega=sqrt(T_Delta*ErrH1Omega.sum())
ErrH1_all_Gamma=sqrt(T_Delta*ErrH1Gamma.sum())
print([ErrL2_all_Omega,ErrL2_all_Gamma,ErrH1_all_Omega,ErrH1_all_Gamma])