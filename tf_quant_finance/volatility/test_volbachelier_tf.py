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

import sabr_approx_tf as sabr

import matplotlib.pyplot as plt
import vanilla_n_tf as vp
import volbachelier_tf as vb
import autograd.numpy as np

# SABR parameters
f = 1.0 #0.00434015
alpha_org = 0.16575423
beta_org = .6#0.7#0.16415365
nu_org = 0.0632859
rho_org =  -0.32978014
T =  5

displacement = 0 #0.005
kmin = .25
kmax = 5
kval = np.arange(kmin, kmax, 0.01)

alpha_vec = [0.1, 0.15, 0.175, 0.2, 0.25, 0.3, 0.5, 0.75]
beta_vec = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
rho_vec = [-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
nu_vec = [0.001, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.5, 0.75, 1.0, 1.5]


for alpha in alpha_vec:
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha, beta_org, rho_org, nu_org)
    oval = vp.option_price(f, kval, yval, T, 0)
    ival = vb.volbachelier_tf(1., kval, f, T, oval)
    plt.plot(kval, yval - ival)
plt.title('Bachelier implied vol vs SABR 1')
plt.legend()
plt.show()

for beta in beta_vec:
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta, rho_org, nu_org)
    oval = vp.option_price(f, kval, yval, T, 0)
    ival = vb.volbachelier_tf(1., kval, f, T, oval)
    plt.plot(kval, yval - ival)
plt.title('Bachelier implied vol vs SABR 2')
plt.legend()
plt.show()


for rho in rho_vec:
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta_org, rho, nu_org)
    oval = vp.option_price(f, kval, yval, T, 0)
    ival = vb.volbachelier_tf(1., kval, f, T, oval)
    plt.plot(kval, yval - ival)
plt.title('Bachelier implied vol vs SABR 3')
plt.legend()
plt.show()


for nu in nu_vec:
    yval = sabr.volsabr_h_n_tf(f, kval, T, displacement, alpha_org, beta_org, rho_org, nu)
    oval = vp.option_price(f, kval, yval, T, 0)
    ival = vb.volbachelier_tf(1., kval, f, T, oval)
    plt.plot(kval, yval - ival)
plt.title('Bachelier implied vol vs SABR 4')
plt.legend()
plt.show()