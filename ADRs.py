# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 15:46:39 2022
ADR equations
@author: willg
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import flopy
#import gstools as gs

import scipy
from scipy import optimize
from scipy.special import erfc as erfc
import math
from math import pi
from matplotlib import rc
import matplotlib.ticker as mtick
from labellines import labelLine, labelLines
rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams['font.size'] = 16
#%matplotlib inline
#%% Analytical solution
# Retardation with 1st type BC (equation C5)
# 'u' term identical in equation c5 and c6 (type 3 inlet)
# Write out the name of each of these variables
def ADEwReactions_type1_fun(x, t, v, D, R, gamma, mu, C0, t0, Ci):
    u = v*(1+(4*mu*D/v**2))**(1/2)
    
    # Note that the '\' means continued on the next line
    Atrf = np.exp(-mu*t/R)*(1- (1/2)* \
        erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
        (1/2)*np.exp(v*x/D)*erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
    
    # term with B(x, t)
    Btrf = 1/2*np.exp((v-u)*x/(2*D))* \
        erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) \
        + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/ \
        (2*(D*R*t)**(1/2)))
    
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        
        indices_below_zero = tt0 <= 0
        # set values equal to 1 (but this could be anything)
        tt0[indices_below_zero] = 1
    
        Bttrf = 1/2*np.exp((v-u)*x/(2*D))* \
            erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) \
            + 1/2*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/ \
            (2*(D*R*tt0)**(1/2)))
        
        # Now set concentration at those negative times equal to 0
        Bttrf[indices_below_zero] = 0
        if mu >0:
            C_out = (gamma/mu)+ (Ci- gamma/mu)*Atrf + \
                (C0 - gamma/mu)*Btrf - C0*Bttrf
        else:
            C_out = Ci*Atrf + C0 *Btrf - C0*Bttrf
            
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        if mu >0:
            C_out = (gamma/mu)+ (Ci- gamma/mu)*Atrf + (C0 - gamma/mu)*Btrf;
        else: # if mu = 0 then we would get nans
            C_out = (Ci)*Atrf + (C0)*Btrf
        
    
    # Return the concentration (C) from this function
    return C_out
#%%
# Retardation with 3rd type inlet BC, infinite length (equation C5 in Van Genuchten and Alves)
# NOTE that the zero order terms have been omitted
def ADEwReactions_type3_fun(x, t, v, D, R, gamma, mu, C0, t0, Ci):
    u = v*(1+(4*mu*D/v**2))**(1/2)

    if mu == 0:
        # Infinite length, type 3 inlet  
        Atrv = (1- (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
            (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*D*R*t)) + \
            (1/2)*(1 + v*x/D + v**2*t/(D*R))*np.exp(v*x/D)* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
        # term with B(x, t)
        Btrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) 
        
    else: 
        # Infinite length, type 3 inlet  
        Atrv = np.exp(-mu*t/R)*(1- (1/2)*erfc((R*x - v*t)/(2*(D*R*t)**(1/2))) - \
            (v**2*t/(pi*D*R))**(1/2)*np.exp(-(R*x - v*t)**2/(4*D*R*t)) + \
            (1/2)*(1 + v*x/D + v**2*t/(D*R))*np.exp(v*x/D)* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2))))
        # term with B(x, t)
        Btrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*t)/(2*(D*R*t)**(1/2))) + \
            (v/(v-u))*np.exp((v+u)*x/(2*D))* erfc((R*x + u*t)/(2*(D*R*t)**(1/2))) + \
            (v**2/(2*mu*D))*np.exp((v*x/D) - (mu*t/R))* \
            erfc((R*x + v*t)/(2*(D*R*t)**(1/2)))
    
    # if a pulse type injection
    if t0 > 0:
        tt0 = t - t0
        # 
        indices_below_zero = tt0 <= 0
        
        if np.sum(indices_below_zero) > 0:
            # set values equal to 1 (but this could be anything)
            tt0[indices_below_zero] = 1
        
        if mu == 0:
            Bttrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) 
        else:
            Bttrv = (v/(v+u))*np.exp((v-u)*x/(2*D))* erfc((R*x - u*tt0)/(2*(D*R*tt0)**(1/2))) + \
                (v/(v-u))*np.exp((v+u)*x/(2*D))* erfc((R*x + u*tt0)/(2*(D*R*tt0)**(1/2))) + \
                (v**2/(2*mu*D))*np.exp((v*x/D) - (mu*tt0/R))* \
                erfc((R*x + v*tt0)/(2*(D*R*tt0)**(1/2)))

        # Now set concentration at those negative times equal to 0
        if np.sum(indices_below_zero) > 0:
            Bttrv[indices_below_zero] = 0

        if gamma > 1E-10:
            C_out = gamma/mu + (Ci-gamma/mu)*Atrv + (C0- gamma/mu)*Btrv - C0*Bttrv
        else:
            C_out = Ci*Atrv + C0*Btrv - C0*Bttrv
            
    else: # if a continous injection then ignore the Bttrf term (no superposition)
        if gamma > 1E-10:
            C_out = gamma/mu + (Ci-gamma/mu)*Atrv + (C0- gamma/mu)*Btrv
        else:
            C_out = Ci*Atrv + C0*Btrv
    # Return the concentration (C) from this function
    return C_out
#%% PFAS vadose zone leaching Guo et al https://doi.org/10.1016/j.advwatres.2021.104102
#variable description
#L = depth of the vadose zone
#ws = Damkohler number = ratio between characteristic time scales of transport vs reaction
#aS = first order rate constant for kinetic sorption - as in the paper
#Rs = pbKd/(Sw/phi)
#Raw = KawAwi/(Sw/phi)
#Fs = fraction avaialble 

# def ADE_PFAS_2DomainSPKAwAWIA_type1_fun():
#     T = (v*t)/L
#     T0 = (v*t0)/L
#     Z = z/L
#     P = (v*L)/D
#     Bs = (1+Fs*Rs)/(1+Rs)
#     ws = aS*(1-Bs)*(1+Rs)*(L/v)
    
#     AZT = (1/2)*erfc((RZ-T))
#     BZT
#     a1
#     b1
#     Ja1b1
#     gZt #for type 3 this term is different
    
 
    
 
#Guo et al Appendix - Special case of the analytical solution: Equilibrium adsorption
#Solid-phase sorption assumed equailibrium: where Fs = 1, and Cs2 = 0
#only works for concentration profiles at the moment
def ADE_PFASwEqSorption_type1_fun(v, t, L, D, R, z, t0, C0):
    T = (v*t)/L
    T0 = (v*t0)/L
    Z = z/L
    P = (v*L)/D
    
    AZT = (1/2)*erfc((R*Z - T)/(2*(T*R/P)**(1/2))) + ((T*P)/(pi*R))**(1/2)*np.exp((-(R*Z - T)**2)/(4*T*R/P)) - \
        (1/2)*(1 + (P*Z) + ((P*T)/R))*np.exp(P*Z)*erfc((R*Z + T)/(2*(T*R/P)**(1/2)))
        
    AZTT0 = (1/2)*erfc((R*Z - (T-T0)/(2*((T-T0)*R/P)**(1/2)))) + (((T-T0)*P)/(pi*R))**(1/2)*np.exp((-(R*Z - (T-T0))**2)/(4*(T-T0)*R/P)) - \
        (1/2)*(1 + (P*Z) + ((P*(T-T0)/R))*np.exp(P*Z)*erfc((R*Z + (T-T0)))/(2*((T-T0)*R/P)**(1/2)))
    
    if 0 < T <= T0:  #continuous 
        C_out = C0 * AZT
        
    elif T > T0: #pulse
        C_out = (C0*AZT) - (C0*AZTT0)
        
    return C_out
#%%
#cm
def ADE_PFASwEqSorption_typex_fun_test(v, t, L, D, R, z, C0): #no pulse
    T = (v*t)/L #dimless time
    print(T)
    #T0 = (v*t0)/L
    Z = z/L #dimless depth?
    print(Z)
    P = (v*L)/D #dimless diffusion
    print(P)
    AZT = (1/2)*erfc((R*Z - T)/(2*(T*R/P)**(1/2))) + ((T*P)/(pi*R))**(1/2)*np.exp((-(R*Z - T)**2)/(4*T*R/P)) - \
        (1/2)*(1 + (P*Z) + ((P*T)/R))*np.exp(P*Z)*erfc((R*Z + T)/(2*(T*R/P)**(1/2)))
    print(AZT)
    
    C_out = C0 * AZT
        
    return C_out

def ADE_PFASwEqSorption_typex_fun_test2(v, t, L, D, R, z, t0, C0): #pulse
    T = (v*t)/L #dimless time
    print(T)
    T0 = (v*t0)/L
    Z = z/L #dimless depth?
    print(Z)
    P = (v*L)/D #dimless diffusion
    print(P)
    AZT = (1/2)*erfc((R*Z - T)/(2*(T*R/P)**(1/2))) + ((T*P)/(pi*R))**(1/2)*np.exp((-(R*Z - T)**2)/(4*T*R/P)) - \
        (1/2)*(1 + (P*Z) + ((P*T)/R))*np.exp(P*Z)*erfc((R*Z + T)/(2*(T*R/P)**(1/2)))
        
    AZTT0 = (1/2)*erfc((R*Z - (T-T0)/(2*((T-T0)*R/P)**(1/2)))) + (((T-T0)*P)/(pi*R))**(1/2)*np.exp((-(R*Z - (T-T0))**2)/(4*(T-T0)*R/P)) - \
        (1/2)*(1 + (P*Z) + ((P*(T-T0)/R))*np.exp(P*Z)*erfc((R*Z + (T-T0)))/(2*((T-T0)*R/P)**(1/2)))
    
    print(AZTT0)
    
    C_out = C0*AZT - C0*AZTT0
        
    return C_out

v = 1.3*10**-5 #cm/s
t_yrs = 20
yrs_s = 3.154*10**7
ts = t_yrs * yrs_s
t = np.linspace(0, ts, 50) #s
L = 300 #cm
D = 5.4*10**-5 #cm2/s
R = 10 #arbitray spot on diagram
z = 200
t0 = 5*yrs_s
C0 = 100 #mg/L

C_out = ADE_PFASwEqSorption_typex_fun_test(v, t, L, D, R, z, C0)
plt.plot(t/yrs_s, C_out/C0)   

# C_out2 = ADE_PFASwEqSorption_typex_fun_test2(v, t, L, D, R, z, t0, C0)
# plt.plot(t/yrs_s, C_out2/C0) 
