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

# f = 1.0 is sufficient due to scaling property

import volsabr_h_n_tf as sabr

import matplotlib.pyplot as plt
#import numpy as np
import autograd.numpy as np
# SABR parameters
f = 1.0 #0.00434015
alpha_org = 0.16575423
beta_org = .6#0.7#0.16415365
nu_org = 0.0632859
rho_org =  -0.32978014
T = 5
      
displacement = 0 #0.005
kmin = -displacement
kmax = 10
kval = np.arange(kmin, kmax, 0.01)

alpha_vec = [0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5, 0.75, 1., 1.5]
beta_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rho_vec = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
nu_vec = [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 0.75, 1.0, 1.5]
displacement_vec = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]

print('alpha varies')
for alpha in alpha_vec:
    print('alpha:', alpha)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha, beta_org, rho_org, nu_org)
    plt.plot(kval,yval,label=alpha)
plt.title('alpha varies')
plt.legend()
plt.show()


print('beta varies')
for beta in beta_vec:
    print('beta:', beta)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta, rho_org, nu_org)
    plt.plot(kval,yval,label=beta)
plt.title('beta varies')
plt.legend()
plt.show()

print('rho varies')
for rho in rho_vec:
    print('rho:', rho)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta_org, rho, nu_org)
    plt.plot(kval,yval,label=rho)
plt.title('rho varies')
plt.legend()
plt.show()

print('nu varies')
for nu in nu_vec:
    print('nu:' nu)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha, beta, rho_org,nu)
    plt.plot(kval,yval,label=nu)
plt.title('nu varies')
plt.legend()
plt.show()

print('displacement varies')
for displacement in displacement_vec:
    print('displacement:', displacement)
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta_org, rho_org,nu_org)
    plt.plot(kval,yval,label=displacement)
plt.title('displacement varies')
plt.legend()
plt.show()

