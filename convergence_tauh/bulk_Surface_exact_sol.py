from ngsolve import *

def sol_token(t,T_Delta,normal):
    Diff_CF=1/(8*pi**2)
    Delta=1
    Sigma=1
    Diff_Gamma_CF=1/(4*pi**2)
    Delta_Gamma=1
    k=1
    ##
    Phi=cos(2*pi*x)*cos(2*pi*y)*exp(-t)
    #Mu=cos(2*pi*x)*cos(2*pi*y)*exp(-t)
    #F_righ1=Mu-(8*pi**2)*Delta*Sigma*cos(2*pi*x)*cos(2*pi*y)*exp(-t)-1/Delta*Sigma*(Phi**3-Phi)
    Mu1=cos(2*pi*x)*cos(2*pi*y)
    Phi1=cos(2*pi*x)*cos(2*pi*y)
    Int_F=Mu1*(exp(-(t-T_Delta))-exp(-t))\
          -(8*pi**2)*Delta*Sigma*cos(2*pi*x)*cos(2*pi*y)*(exp(-(t-T_Delta))-exp(-t))\
          -1/Delta*Sigma*(Phi1**3*(1/3*(exp(-3*(t-T_Delta))-exp(-3*t)))-Phi1*(exp(-(t-T_Delta))-exp(-t)))
    Mu_Gamma=cos(2*pi*x)*cos(2*pi*y)*exp(-t)

    #G_right1=Mu_Gamma-(4*pi**2)*Delta_Gamma*k*cos(2*pi*x)*cos(2*pi*y)*exp(-t)-1/Delta_Gamma*(Phi**3-Phi)-Delta*Sigma*Phi.Diff(x)*normal[0]-Delta*Sigma*Phi.Diff(y)*normal[1]
    Mu_Gamma1=cos(2*pi*x)*cos(2*pi*y)
    Int_G=Mu_Gamma1*(exp(-(t-T_Delta))-exp(-t))\
    -(4*pi**2)*Delta_Gamma*k*cos(2*pi*x)*cos(2*pi*y)*(exp(-(t-T_Delta))-exp(-t))\
    -1/Delta_Gamma*(Phi1**3*(1/3*(exp(-3*(t-T_Delta))-exp(-3*t)))-Phi1*(exp(-(t-T_Delta))-exp(-t)))\
    -(Delta*Sigma*Phi1.Diff(x)*normal[0]+Delta*Sigma*Phi1.Diff(y)*normal[1])*(exp(-(t-T_Delta))-exp(-t))
    
    F_Bulk_right=CF(1/T_Delta*Int_F.Compile())
    G_Gamma_right=CF(1/T_Delta*Int_G.Compile())
    Phi_0=CF(cos(2*pi*x)*cos(2*pi*y))
    Phi_ex=CF(Phi)
    return Diff_CF,Delta,Sigma,Diff_Gamma_CF,Delta_Gamma,k,Phi_0,Phi_ex,F_Bulk_right,G_Gamma_right