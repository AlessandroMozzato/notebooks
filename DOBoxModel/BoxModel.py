
# coding: utf-8

# In[1]:

from scipy.io import netcdf
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import sys
import math
from pylab import *
from IPython.display import display, Math, Latex
from numba import jit
import glob


# In[135]:

# Model from Johnson and Marshal 2009 plus temperature
# The model uses a Time discretization with a 3rd order Adam Bashforth scheme


# In[136]:

# Initial values for variables
D_0 = 500 ;
S_t0 = 35 ;
S_d0 = 35 ;
S_n0 = 35 ;
S_s0 = 35 ;
T_t0 = 25 + 273 ;
T_d0 = 5 + 273;
T_n0 = 5 + 273;
T_s0 = 5 + 273;


# In[137]:

# Time step values
n_it = 2000000; # number of iterations
Tott = 10000 ; #total time (in years)
step = 1000 ;  
sav = n_it / step ;
Tott_s = Tott*365*24*60*60; #total time in seconds
h    = Tott_s / n_it ; #timestep
timestepday = h / (24*60*60) ;
h_for_year = 365*24*60*60 / h ;


# In[138]:

# Parameters set up

# Freshwater forcing (with possible Stochastic forcing)
stochasticE = "NO" ; # can be YES or NO
sigmaE = 0.4;
muE = 10**6;

if stochasticE=="YES":
    E1 = 1*sigmaE*randn(n_it+1,1)+muE ;
    E2 = 1*sigmaE*randn(n_it+1,1)+muE ;
    E3 = 1*sigmaE*randn(n_it+1,1)+muE ;
else:
    E1 = 0.8*(10**6)*ones(n_it+1) ;
    E2 = 0.8*(10**6)*ones(n_it+1) ;
    E3 = 0.8*(10**6)*ones(n_it+1) ;
    
# Other Parameters
A_G  = 2000 ; #eddy diffusion parameter
tau  = 0.2; #zonal wind stress  
kappa = 2*10**(-5); #constant dyapicnal coefficient
#E    = 0.8*10^6 ; %freshwater flux
r    = 8*10**6 ;
   
#temperature relaxation coefficient   
eta_t = 1e-07 ;
eta_n = 1e-07 ;
eta_s = 1e-07 ;
   
delta = 0 ; 



# In[139]:

#Parameter fixed
A    = 2.6*10**14; # m2 Area where thermocline takes place
L_x  = 3*10**7 ; # m zonal extent
L_y  = 10**6; # m meridional extent
rho_0 = 1027.5 ; # kg/m3 reference density
f_S  = -10**(-4) ; # s-1 Coriolis parameter
f_N  = 10**(-4) ; # Coriolis parameter
g    = 9.8 ; # gravity acceleration
V_n  = 3*10**15 ; # volume of high latitude northern box
V_s  = 9*10**15 ; # volume southern ocean box
V_tot = 1.2*10**18 ; # Total volume of the ocean
S_0 = 35 ; # Reference salinity
Ttar_t  = 25 + 273; # Target temperature for restoration
Ttar_s  = 5 + 273;
Ttar_n  = 5 + 273;
T_0  = 5 + 273; # Reference temperature 
alpha = 0.0002 ; # temperature density expansion 
beta = 0.0008 ; # salinity density expansion


# In[140]:

# Variable initialisation (initializing variables where results are going to be stored)
kk = 1 ;
   
D = np.zeros(sav+1);
T_t = zeros(sav+1);
T_n = zeros(sav+1);
T_d = zeros(sav+1);
T_s = zeros(sav+1);
  
S_t = zeros(sav+1);
S_n = zeros(sav+1);
S_d = zeros(sav+1);
S_s = zeros(sav+1);
rho_t = zeros(sav+1);
rho_n = zeros(sav+1);
rho_d = zeros(sav+1);
rho_s = zeros(sav+1);
   
q_Ek_v = zeros(sav+1) ;
q_Eddy_v = zeros(sav+1) ;
q_S_v = zeros(sav+1) ;
q_U_v = zeros(sav+1) ;
q_N_v = zeros(sav+1) ;
   
rho_t_t = rho_0*(1- alpha*(T_t0 - T_0) + beta*(S_t0 - S_0)) ; 
rho_n_t = rho_0*(1- alpha*(T_n0 - T_0) + beta*(S_n0 - S_0)) ;
rho_d_t = rho_0*(1- alpha*(T_d0 - T_0) + beta*(S_d0 - S_0)) ;


# In[141]:

# Calculating f(x_0) for the first iteration
   
q_Ek     =  tau*L_x / (rho_0*abs(f_S)) ;
q_Eddy   =  A_G*D_0*L_x / L_y ; 
q_S      =  q_Ek - q_Eddy ;
q_U      =  kappa*A / D_0 ;   
q_N      =  g*(rho_d_t-rho_t_t)*D_0**2 / (2*f_N*rho_0) ;

V_t0     =  A*D_0 ; 
V_d0     =  V_tot - V_t0 - V_n - V_s ;
   
if q_S > 0 :
    Stop = S_s0 ;
    Sbot = S_d0 ;
    Ttop = T_s0 ;
    Tbot = T_d0 ;
else :
    Stop = S_t0 ;
    Sbot = S_s0 ;
    Ttop = T_t0 ;
    Tbot = T_s0 ;
   
# Deciding if there's convection in the northern box
if rho_d_t- rho_n_t > delta :
     q_N = 0 ;  
   
f1_0 = q_Ek - q_Eddy + q_U - q_N ;
f2_0 = q_U*S_d0 + 2*E1[0]*S_0 - q_N*S_t0 + q_S*Stop + r*(S_n0 - S_t0) ; 
f3_0 = -q_N*S_n0 + q_N*S_t0 - E2[0]*S_0 + r*(S_t0 - S_n0) ;
f4_0 = q_N*S_n0 - q_U*S_d0 - q_S*Sbot ;
f5_0 = q_S*Sbot - E3[0]*S_0 - q_S*Stop;

# Equation for the temperature balance
   
f6_0 = q_U*T_d0 + eta_t*V_t0*(Ttar_t - T_t0) - q_N*T_t0 + q_S*Ttop + r*(T_n0 - T_t0) ;
f7_0 = -q_N*T_n0 + q_N*T_t0 + eta_n*V_n*(Ttar_n - T_n0) + r*(T_t0 - T_n0) ;
f8_0 = q_N*T_n0 - q_U*T_d0 - q_S*Tbot ;
f9_0 = q_S*Tbot + eta_s*V_s*(Ttar_s - T_s0) - q_S*Ttop ;
   
#Equation for the thermocline
   
D_1 =   h*(q_Ek - q_Eddy + q_U - q_N) / A + D_0;
V_d1    = V_tot - A*D_1 - V_n - V_s ;
   
# Equations for the salinity budget
   
S_t1 = h*(q_U*S_d0 + 2*E1[0]*S_0 - q_N*S_t0 + q_S*Stop + r*(S_n0 - S_t0)) / (D_1*A) + S_t0*D_0/(D_1) ; 
S_n1 = h*(-q_N*S_n0 + q_N*S_t0 - E2[0]*S_0 + r*(S_t0 - S_n0)) / V_n + S_n0 ;
S_d1 = h*(q_N*S_n0 - q_U*S_d0 - q_S*Sbot) / (V_d1) + S_d0*V_d0/V_d1 ;
S_s1 = h*(q_S*Sbot - E3[0]*S_0 - q_S*Stop) / V_s + S_s0 ;
   
# Equations for the temperature balance
   
T_t1 = h*(q_U*T_d0 + eta_t*A*D_1*(Ttar_t - T_t0) - q_N*T_t0 + q_S*Ttop + r*(T_n0 - T_t0)) / (D_1*A) + T_t0*D_0/(D_1) ; 
T_n1 = h*(-q_N*T_n0 + q_N*T_t0 + eta_n*V_n*(Ttar_n - T_n0) + r*(T_t0 - T_n0)) / V_n + T_n0 ;
T_d1 = h*(q_N*T_n0 - q_U*T_d0 - q_S*Tbot) / (V_d1) + T_d0*V_d0/V_d1 ;
T_s1 = h*(q_S*Tbot + eta_s*V_s*(Ttar_s - T_s0) - q_S*Ttop) / V_s + T_s0 ;
   
# Density calculation
   
rho_t_t = rho_0*(1 - alpha*(T_t1 - T_0) + beta*(S_t1 - S_0)) ;
rho_d_t = rho_0*(1 - alpha*(T_d1 - T_0) + beta*(S_d1 - S_0)) ;
#rho_s_t = rho_0*(1 - alpha*(T_s1 - T_0) + beta*(S_s1 - S_0)) ;
rho_n_t = rho_0*(1 - alpha*(T_n1 - T_0) + beta*(S_n1 - S_0)) ;
   
# Saving variables
   
D_0 = D_1 ;
T_t0 = T_t1 ;
T_n0 = T_n1 ;
T_d0 = T_d1 ;
T_s0 = T_s1 ;
   
S_t0 = S_t1 ;
S_n0 = S_n1 ;
S_d0 = S_d1 ;
S_s0 = S_s1 ;


# In[142]:

# Calculating f(x_0) for the first iteration (second bit)
   
q_Ek     =  tau*L_x / (rho_0*abs(f_S)) ;
q_Eddy   =  A_G*D_0*L_x / L_y ; 
q_S      =  q_Ek - q_Eddy ;
q_U      =  kappa*A / D_0 ;
q_N      =  g*(rho_d_t-rho_t_t)*D_0**2 / (2*f_N*rho_0) ;
   
if q_S > 0 :
    Stop = S_s0 ;
    Sbot = S_d0 ;
    Ttop = T_s0 ;
    Tbot = T_d0 ;
else :
    Stop = S_t0 ;
    Sbot = S_s0 ;
    Ttop = T_t0 ;
    Tbot = T_s0 ;
 
V_t0     =  A*D_0 ; 
V_d0     =  V_tot - V_t0 - V_n - V_s ;
   
# Deciding if there's convection in the northern box
if rho_d_t- rho_n_t > delta :
     q_N = 0 ;  
     
# Equations ofr the thermocline and the salinity balance
     
f1_1 = q_Ek - q_Eddy + q_U - q_N ;
f2_1 = q_U*S_d0 + 2*E1[1]*S_0 - q_N*S_t0 + q_S*Stop + r*(S_n0 - S_t0) ; 
f3_1 = -q_N*S_n0 + q_N*S_t0 - E2[1]*S_0 + r*(S_t0 - S_n0) ;
f4_1 = q_N*S_n0 - q_U*S_d0 - q_S*Sbot ;
f5_1 = q_S*Sbot - E3[1]*S_0 - q_S*Stop;
   
# Equation for the temperature balance
   
f6_1 = q_U*T_d0 + eta_t*A*D_1*(Ttar_t - T_t0) - q_N*T_t0 + q_S*Ttop + r*(T_n0 - T_t0);
f7_1 = -q_N*T_n0 + q_N*T_t0 + eta_n*V_n*(Ttar_n - T_n0) + r*(T_t0 - T_n0);
f8_1 = q_N*T_n0 - q_U*T_d0 - q_S*Tbot ;
f9_1 = q_S*Tbot + eta_s*V_s*(Ttar_s - T_s0) - q_S*Ttop;
   
# Equation for the thermocline
   
D_1 =   h*(q_Ek - q_Eddy + q_U - q_N) / A + D_0;
V_d1    = V_tot - A*D_1 - V_n - V_s ;
   
# Equations for the salinity budget
   
S_t1 = h*(q_U*S_d0 + 2*E1[1]*S_0 - q_N*S_t0 + q_S*Stop + r*(S_n0 - S_t0)) / (D_1*A) + S_t0*D_0/(D_1) ; 
S_n1 = h*(-q_N*S_n0 + q_N*S_t0 - E2[1]*S_0 + r*(S_t0 - S_n0)) / V_n + S_n0 ;
S_d1 = h*(q_N*S_n0 - q_U*S_d0 - q_S*Sbot) / (V_d1) + S_d0*V_d0/V_d1 ;
S_s1 = h*(q_S*Sbot - E3[1]*S_0 - q_S*Stop) / V_s + S_s0 ;
   
# Equations for the temperature balance
   
T_t1 = h*(q_U*T_d0 + eta_t*A*D_1*(Ttar_t - T_t0) - q_N*T_t0 + q_S*Ttop + r*(T_n0 - T_t0)) / (D_1*A) + T_t0*D_0/(D_1) ; 
T_n1 = h*(-q_N*T_n0 + q_N*T_t0 + eta_n*V_n*(Ttar_n - T_n0) + r*(T_t0 - T_n0)) / V_n + T_n0 ;
T_d1 = h*(q_N*T_n0 - q_U*T_d0 - q_S*Tbot) / (V_d1) + T_d0*V_d0/V_d1 ;
T_s1 = h*(q_S*Tbot + eta_s*V_s*(Ttar_s - T_s0) - q_S*Ttop) / V_s + T_s0 ;
  
# Saving variables
   
D_0 = D_1 ;
T_t0 = T_t1 ;
T_n0 = T_n1 ;
T_d0 = T_d1 ;
T_s0 = T_s1 ;
   
S_t0 = S_t1 ;
S_n0 = S_n1 ;
S_d0 = S_d1 ;
S_s0 = S_s1 ;

total = range(3,n_it+1)
for k in total :
        
#Density calculation
   
   rho_t_t = rho_0*(1 - alpha*(T_t1 - T_0) + beta*(S_t1 - S_0)) ;
   rho_d_t = rho_0*(1 - alpha*(T_d1 - T_0) + beta*(S_d1 - S_0)) ;
   #rho_s_t = rho_0*(1 - alpha*(T_s1 - T_0) + beta*(S_s1 - S_0)) ;
   rho_n_t = rho_0*(1 - alpha*(T_n1 - T_0) + beta*(S_n1 - S_0)) ;
       
   q_Ek     =  tau*L_x / (rho_0*abs(f_S)) ;
   q_Eddy   =  A_G*D_0*L_x / L_y ; 
   q_S      =  q_Ek - q_Eddy ;
   q_U      =  kappa*A / D_0 ;
   q_N      =  g*(rho_d_t-rho_t_t)*D_0**2 / (2*f_N*rho_0) ;
   
   # Updating values of f(x_n+1)
   
   if q_S > 0 :
       Stop = S_s0 ;
       Sbot = S_d0 ;
       Ttop = T_s0 ;
       Tbot = T_d0 ;
   else :
       Stop = S_t0 ;
       Sbot = S_s0 ;
       Ttop = T_t0 ;
       Tbot = T_s0 ;
      
   # Deciding if there's convection in the northern box
   
   if rho_d_t - rho_n_t > delta :
         q_N = 0 ;  
                    
   f1_2 = q_Ek - q_Eddy + q_U - q_N ;
   f2_2 = q_U*S_d0 + 2*E1[k]*S_0 - q_N*S_t0 + q_S*Stop + r*(S_n0 - S_t0) ; 
   f3_2 = -q_N*S_n0 + q_N*S_t0 - E2[k]*S_0 + r*(S_t0 - S_n0) ;
   f4_2 = q_N*S_n0 - q_U*S_d0 - q_S*Sbot ;
   f5_2 = q_S*Sbot - E3[k]*S_0 - q_S*Stop;
   
   # Equation for the temperature balance
   
   f6_2 = q_U*T_d0 + eta_t*A*D_1*(Ttar_t - T_t0) - q_N*T_t0 + q_S*Ttop + r*(T_n0 - T_t0) ;
   f7_2 = -q_N*T_n0 + q_N*T_t0 + eta_n*V_n*(Ttar_n - T_n0) + r*(T_t0 - T_n0) ;
   f8_2 = q_N*T_n0 - q_U*T_d0 - q_S*Tbot ;
   f9_2 = q_S*Tbot + eta_s*V_s*(Ttar_s - T_s0) - q_S*Ttop ;
   
   V_d0     =  V_d1 ;
   
   # Calculation of x_n+2 with the 3-step Adams-Bashforth
   
   D_1  = D_0               + h/12*( 23*f1_2 - 16*f1_1 + 5*f1_0 )/A ;   

   V_t1     =  A*D_1 ; 
   V_d1     =  V_tot - V_t1 - V_n - V_s ;

   S_t1 = S_t0*D_0/D_1      + h/12*( 23*f2_2 - 16*f2_1 + 5*f2_0 )/V_t1 ;
   S_n1 = S_n0              + h/12*( 23*f3_2 - 16*f3_1 + 5*f3_0 )/V_n ;
   S_d1 = S_d0*V_d0/V_d1    + h/12*( 23*f4_2 - 16*f4_1 + 5*f4_0 )/V_d1 ;
   S_s1 = S_s0              + h/12*( 23*f5_2 - 16*f5_1 + 5*f5_0 )/V_s ;
   
   T_t1 = T_t0*D_0/D_1      + h/12*( 23*f6_2 - 16*f6_1 + 5*f6_0 )/V_t1 ;
   T_n1 = T_n0              + h/12*( 23*f7_2 - 16*f7_1 + 5*f7_0 )/V_n ;
   T_d1 = T_d0*V_d0/V_d1    + h/12*( 23*f8_2 - 16*f8_1 + 5*f8_0 )/V_d1 ;
   T_s1 = T_s0              + h/12*( 23*f9_2 - 16*f9_1 + 5*f9_0 )/V_s ;

   
   # Saving variables
   
   if (k % step)==0 :   
       D[kk] = D_1 ;       
   
       T_t[kk] = T_t1 ;    
       T_n[kk] = T_n1 ;   
       T_d[kk] = T_d1 ;   
       T_s[kk] = T_s1 ;  
   
       S_t[kk] = S_t1 ;  
       S_n[kk] = S_n1 ;  
       S_d[kk] = S_d1 ;   
       S_s[kk] = S_s1 ;
   
       q_Ek_v[kk] = q_Ek ;
       q_Eddy_v[kk] = q_Eddy ;
       q_S_v[kk] = q_S ;
       q_U_v[kk] = q_U ;
       q_N_v[kk] = q_N ;
   
       rho_t[kk] = rho_t_t ; 
       rho_d[kk] = rho_d_t ;
       rho_s[kk] = rho_0*(1 - alpha*(T_s1 - T_0) + beta*(S_s1 - S_0)) ;
       rho_n[kk] = rho_n_t ; 
   
       kk= kk + 1 ;

print kk   
print k


# In[143]:

subplot(2,3,1)
plt.plot(D) #Use b2r colourmap
title("Thermocline")
#subplots_adjust(right=2.3)
#colorbar()
subplot(2,3,2)
plt.plot(S_t)
plt.plot(S_d)
plt.plot(S_n)
plt.plot(S_s)
title("Salinity")

subplot(2,3,3)
plt.plot(T_t-273)
plt.plot(T_d-273)
plt.plot(T_n-273)
plt.plot(T_s-273)
title("Temperature")

subplot(2,3,4)
plt.plot(rho_t-1000)
plt.plot(rho_d-1000)
plt.plot(rho_n-1000)
plt.plot(rho_s-1000)
title("Density")

subplot(2,3,5)
plt.plot(E1/10**6)

subplot(2,3,6)
plt.plot(q_U_v/10**6)
plt.plot(q_Eddy_v/10**6)
plt.plot(q_S_v/10**6)
plt.plot(q_N_v/10**6)



subplots_adjust(right=2.5,top=2)


# In[143]:




# In[143]:



