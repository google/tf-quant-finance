# Lint as: python3
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sabr_approx_tf as sabr
import bachelier_tf as vptf
import volbachelier_tf as bvtf

import numpy as np
import matplotlib.pyplot as plt

# SABR parameters
# SABR parameters
f = 1.0 #0.00434015
alpha_org = 0.16575423
beta_org = .6#0.7#0.16415365
nu_org = 0.2632859
rho_org =  -0.32978014
T = 5
      
displacement_org = 0. #0.005
kmin = -displacement_org
kmax = 10
kval = np.arange(kmin, kmax, 0.01)
kval[0] = (kval[0] + kval[1])/2
vol = np.zeros(len(kval))     

alpha_vec = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5, 0.75, 1., 1.5]
beta_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rho_vec = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
nu_vec = [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 0.75, 1.0, 1.5]
displacement_vec = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]

print('alpha varies')
for alpha in alpha_vec:
    print('alpha: ', alpha)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha, beta_org, rho_org,nu_org)
    cval = vptf.bachelier_option_price(f,kval,yval,T,0.)
    yval1 = bvtf.volbachelier_tf(1, kval, f, T, cval)
    label1 = 'approx ' + str(alpha)
    label2 = 'iv ' + str(alpha)
    plt.plot(kval,yval,label= label1)
    plt.plot(kval, yval1, label=label2)
plt.title('alpha varies')
plt.legend()
plt.show()


print('beta varies')
for beta in beta_vec:
    print('parameters: ', beta)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha_org, beta, rho_org,nu_org)
    cval = vptf.bachelier_option_price(f,kval,yval,T,0.)
    yval1 = bvtf.volbachelier_tf(1, kval, f, T, cval)
    label1 = 'approx ' + str(beta)
    label2 = 'iv ' + str(beta)
    plt.plot(kval,yval,label= label1)
    plt.plot(kval, yval1, label=label2)
plt.title('beta varies')
plt.legend()
plt.show()

print('rho varies')
for rho in rho_vec:
    print('parameters: ', rho)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha_org, beta_org, rho,nu_org)
    cval = vptf.bachelier_option_price(f,kval,yval,T,0.)
    yval1 = bvtf.volbachelier_tf(1, kval, f, T, cval)
    label1 = 'approx ' + str(rho)
    label2 = 'iv ' + str(rho)
    plt.plot(kval,yval,label= label1)
    plt.plot(kval, yval1, label=label2)
plt.title('rho varies')
plt.legend()
plt.show()

print('nu varies')
for nu in nu_vec:
    print('parameters: ', nu)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha_org, beta_org, rho_org,nu)
    cval = vptf.bachelier_option_price(f,kval,yval,T,0.)
    yval1 = bvtf.volbachelier_tf(1, kval, f, T, cval)
    label1 = 'approx ' + str(nu)
    label2 = 'iv ' + str(nu)
    plt.plot(kval,yval,label= label1)
    plt.plot(kval, yval1, label=label2)
plt.title('nu varies')
plt.legend()
plt.show()

print('displacement varies')
for displacement in displacement_vec:
    print('parameters: ', displacement)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha_org, beta_org, rho_org,nu_org)
    label1 = 'approx ' + str(displacement)
    plt.plot(kval,yval,label= label1)
plt.title('displacement varies')
plt.legend()
plt.show()

# different approximation techniques for SABR and Mean Reverting SABR
kappa = 0.5
cap = 3.

yval1 = sabr.volsabr_h_n_tf(f, kval, T, displacement_org, alpha_org, beta_org, rho_org,nu_org)
yval2 = sabr.volsabr_mr_n_tf(f,kval,T,displacement_org, alpha_org, beta_org, rho_org, nu_org, kappa)
yval3 = sabr.volsabr_h_n_cap_tf(f,kval,T,displacement_org, alpha_org, beta_org, rho_org, nu_org, cap)

label1 = 'Hagan approx ' 
label2 = 'MR SABR approx ' 
label3 = 'Capped SABR approx ' 

plt.plot(kval,yval1,label= label1)
plt.plot(kval,yval2,label= label2)
plt.plot(kval,yval3,label= label3)
plt.title('different SABR approximation')
plt.legend()
plt.show()



