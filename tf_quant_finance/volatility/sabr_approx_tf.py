# -*- coding: utf-8 -*-
"""
""
Created on Fri Nov 22 15:22:13 2019

'Copyright 2020 Joerg Kienitz

'Redistribution and use in source and binary forms, with or without modification,
'are permitted provided that the following conditions are met:

'1. Redistributions of source code must retain the above copyright notice,
'this list of conditions and the following disclaimer.

'2. Redistributions in binary form must reproduce the above copyright notice,
'this list of conditions and the following disclaimer in the documentation
'and/or other materials provided with the distribution.

'3. Neither the name of the copyright holder nor the names of its contributors
'may be used to endorse or promote products derived from this software without
'specific prior written permission.

'THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
'"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
'THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
'ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
'FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
'(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
'LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
'ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
'OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
'THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@author: Joerg Kienitz
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# straight fwd implementation of the Bachelier pricing
# there is a version with just one call to exp !!

def volsabr_h_n_tf(
        forwards, 
        strikes, 
        expiries, 
        displacements, 
        alpha, 
        beta, 
        rho, 
        nu,
        name = None):
    """
    Standard Hagan SABR approximation for the Bachelier/Normal volatility
    
    parameters:
        forwards - forward rate
        strikes - strike as array
        expiries - maturity
        displacements - displacement
        alpha - SABR parameter (initial vol)
        beta - SABR parameter (CEV coefficient)
        rho - SABR parameter (correlation)
        nu - SABR parameter (vol of vol)
        
    For SABR we can always use f=1 and apply the scaling:
    
    if f = f0
    knew -> k/f
    alphanew -> f**(beta-1) * alpha
    ivol1 = sabrapprox.volsabr_h_n(f, kval, T, displacement, alpha, beta, rho, nu)
    price1 = vp.vanilla_n(f,kval,implVolApprox1,0,0,T,1)

    ivol2 = f * sabrapprox.volsabr_h_n(1, knew, T, displacement, alphanew, beta, rho, nu)
    ivol3 = sabrapprox.volsabr_h_n(1, knew, T, displacement, alphanew, beta, rho, nu)
    price2 = vp.vanilla_n(f,kval,ivol2,0,0,T,1)
    price3 = f * vp.vanilla_n(1,kvalnew,ivol3,0,0,T,1)
    
    price1 = price2 = price3
    ivol1 = ivol2 =  f * ivol3
    """
    with tf.compat.v1.name_scope(
      name,
      default_name='sabr_implied_vol_hagan',
      values=[
          forwards, strikes, expiries, displacements, alpha, beta, rho, nu
      ]):
        # conversion maybe outside function!!!!
        forwards = tf.convert_to_tensor(forwards, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=tf.float64, name='expiries')
        displacements = tf.convert_to_tensor(displacements, dtype=tf.float64, name='displacement')
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float64, name='alpha')
        beta = tf.convert_to_tensor(beta, dtype=tf.float64, name='beta')
        rho = tf.convert_to_tensor(rho, dtype=tf.float64, name='rho')
        nu = tf.convert_to_tensor(nu, dtype=tf.float64, name='nu')
        
        
        # identify ATM and non ATM strikes
        eps = tf.constant(1.0e-06, dtype = tf.float64)                           # small number
        index_natm = (tf.math.abs(forwards-strikes) > eps)   # itm/otm strikes 
    
        # case rho = 1 may cause divide by zero problems
        # case rho = 1 may cause divide by zero problems
        # for rho == 1
        # xik = log(2) * log(x+1) -> x > -1
        # for rho == -1
        # xik = log(2) * log(x-1) -> x > 1
        # this means only certain strikes are feasible for calculating implied vols!
        # to this end we only consider the SABR model for correlation values
        # between rho = -0.99 and rho = 0.99
        rho = tf.where(rho == 1., 0.999,tf.where(rho == -1., -0.999, rho))
         
        betam = tf.constant(1.0 - beta)              # often used
    
        fa = forwards+displacements                       # account for displacement for forward
        ka = strikes+displacements                       # account for displacement for strikes
          
        # different cases due to normal, cev, log-normal    
        if 0. < beta and beta < 1.:    # case of true CEV SABR
            gk = tf.zeros_like(strikes)
            gk = tf.where(index_natm,(beta**2 -2.*beta)/24. 
                                      * fa**(-betam)*ka**(-betam)*alpha**2, gk)
            xik = nu / alpha  * (fa**betam - ka**betam) / betam
            xxik = tf.math.log((tf.math.sqrt(1.0- 2.0 * rho * xik + xik**2) - rho + xik) 
                  / (1 - rho))/nu
            vol = tf.where(index_natm, 
                           (fa - ka) / xxik  * (1. + (gk + 0.25 * rho * nu * alpha * beta * fa**(0.5*(beta-1.))*ka**(0.5*(beta-1.)) 
                                                      + (2. - 3. * rho**2) / 24 * nu**2) * expiries), 
                           alpha * fa**beta * (1 + (beta*(beta-2.) * alpha**2 / 24. / fa**(2.*betam)  
                                         + 0.25 * rho*nu*alpha*beta / fa**(betam) 
                                         + (2.-3.*rho**2)/24. * nu**2)*expiries) )          
            return vol
            
        elif beta == 0.:              # case of a Gaussian SV model (normal SABR)
            xik = nu / alpha * (fa - ka)
            xxik = tf.math.log((tf.math.sqrt(1.0- 2.0 * rho * xik + xik**2) - rho + xik) / (1-rho))
            vol = tf.where(index_natm, alpha * xik / xxik * (1 + (1 / 24 * (2 - 3 * rho**2) * nu ** 2) * expiries), 
                           alpha * (1+ (2.-3.*rho**2)/24.*nu**2*expiries))
            return vol
        
        else:                          # case of log-normal SV model (log-normal SABR)
            gk = - 1. / 24. * alpha**2
            xik = nu / alpha * tf.math.log(fa / ka) 
            sum2 = 0.25 * rho * nu * alpha
            xxik = tf.math.log((tf.math.sqrt(1.0- 2.0 * rho * xik + xik**2) - rho + xik) 
                  / (1. - rho))/nu
            vol = tf.where(index_natm, 
                           (fa - ka) / xxik * (1 + (gk + sum2 + 1. / 24. * (2. - 3.*rho**2) * nu**2) * expiries), 
                           alpha * fa * (1.+ (gk + sum2 + (2.-3.*rho**2)/24.*nu**2)*expiries)) 
            return vol
 

def volsabr_mr_n_tf(forwards, strikes, expiries, displacements, alpha, beta, rho, nu, kappa, name = None):
    """
    ' f is the forward
    ' k is the strike
    ' t is maturity
    ' a is displacement
    ' alpha is ATM vol -> sigma in Hagan paper
    ' beta is cev coefficient
    ' rho is correlation
    ' nu is vol of vol
    ' kappa is mean reversion
    """
    with tf.compat.v1.name_scope(
      name,
      default_name='sabr_implied_vol_hagan',
      values=[
          forwards, strikes, expiries, displacements, alpha, beta, rho, nu
      ]):
        # conversion maybe outside function!!!!
        forwards = tf.convert_to_tensor(forwards, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=tf.float64, name='expiries')
        displacements = tf.convert_to_tensor(displacements, dtype=tf.float64, name='displacement')
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float64, name='alpha')
        beta = tf.convert_to_tensor(beta, dtype=tf.float64, name='beta')
        rho = tf.convert_to_tensor(rho, dtype=tf.float64, name='rho')
        nu = tf.convert_to_tensor(nu, dtype=tf.float64, name='nu')
        kappa = tf.convert_to_tensor(kappa, dtype=tf.float64, name='kappa')
        
    
        if kappa > 0.:
            bbar = 2. * rho * nu / alpha * (kappa * expiries - 1. 
                                       + tf.math.exp(-kappa * expiries)) / (kappa ** 2 * expiries ** 2)
            cbar = 1.5 * nu ** 2 / alpha ** 2. * (1. + rho ** 2) * (1. + 2. * kappa * expiries - (2. - tf.math.exp(-kappa * expiries)) ** 2) \
                / (kappa ** 3 * expiries ** 3) + 12. * rho ** 2 * nu ** 2 / alpha ** 2 * (kappa ** 2 * expiries ** 2 * tf.math.exp(-kappa * expiries) 
                - (1. - tf.math.exp(-kappa * expiries)) ** 2) / (kappa ** 4 * expiries ** 4)
            Gstar = (-0.5 * cbar + nu ** 2 / alpha ** 2 * (2. * kappa * expiries - 1. + tf.math.exp(-2. * kappa * expiries)) \
                     / (4. * kappa ** 2 * expiries ** 2)) * alpha ** 2 * expiries

            # std case
            astd = alpha * tf.math.exp(0.5 * Gstar)
            rhostd = bbar / tf.math.sqrt(cbar)
            nustd = alpha * tf.math.sqrt(cbar)
        
            return volsabr_h_n_tf(forwards, strikes, expiries, displacements, astd, beta, rhostd, nustd)
        else:
            return volsabr_h_n_tf(forwards, strikes, expiries, displacements, alpha, beta, rho, nu)


def volsabr_h_n_cap_tf(
        forwards, 
        strikes, 
        expiries, 
        displacements, 
        alpha, 
        beta, 
        rho, 
        nu, 
        cap,
        name = None):
    
    with tf.compat.v1.name_scope(
      name,
      default_name='sabr_implied_vol_hagan',
      values=[
          forwards, strikes, expiries, displacements, alpha, beta, rho, nu
      ]):
        # conversion maybe outside function!!!!
        forwards = tf.convert_to_tensor(forwards, dtype=tf.float64, name='forwards')
        strikes = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
        expiries = tf.convert_to_tensor(expiries, dtype=tf.float64, name='expiries')
        displacements = tf.convert_to_tensor(displacements, dtype=tf.float64, name='displacement')
        alpha = tf.convert_to_tensor(alpha, dtype=tf.float64, name='alpha')
        beta = tf.convert_to_tensor(beta, dtype=tf.float64, name='beta')
        rho = tf.convert_to_tensor(rho, dtype=tf.float64, name='rho')
        nu = tf.convert_to_tensor(nu, dtype=tf.float64, name='nu')
        cap = tf.convert_to_tensor(cap, dtype=tf.float64, name='cap')
    
        eps = 1.0e-06
        index_atm = tf.math.abs(forwards -  strikes)  < eps
    
        vol = tf.zeros_like(strikes)
        fa = forwards + displacements
        ka = strikes + displacements
        
        betam = 1. - beta
        # atm but different cases due to normal, cev, log-normal    
        if 0. < beta and beta < 1.:    # case of true CEV SABR
            volatm = alpha * fa**beta * (1 + (beta*(beta-2.) * alpha**2 / 24. / fa**(2.*betam)  
                                         + 0.25 * rho*nu*alpha*beta / fa**(betam) 
                                         + (2.-3.*rho**2)/24. * nu**2)*expiries)           
            
        elif beta == 0.:              # case of a Gaussian SV model (normal SABR)
            volatm = alpha * (1. + (2.-3.*rho**2)/24.*nu**2*expiries)
        else:                          # case of log-normal SV model (log-normal SABR)
            volatm = alpha * fa * (1.+ (-1. / 24. * alpha**2 + 0.25 * rho * nu * alpha + (2.-3.*rho**2)/24.*nu**2)*expiries)
    
    
        if beta == 1:
            beta = 0.999
    
        rho = tf.where(rho == 1, 0.999, tf.where(rho == -1, -0.999,rho))
        
        # the cap can only be applied if the term under sqrt is positive
        term = tf.math.sqrt(tf.math.maximum(cap ** 2 - 1. + rho**2,0.))

        xip = -rho + term
        xim = -rho - term

        Yp = -tf.math.log((cap - term) / (1. - rho))
        Ym = -tf.math.log((cap + term) / (1. - rho))
    
        # here we need ATM consideration 
        ic = ((ka) ** (1. - beta) - (fa) ** (1. - beta)) / (1. - beta)

        f0 = 0.5 * ((forwards + strikes) + displacements)  # 2* displace?

        gamma = beta * f0 ** (beta-1.)

        Delta0 = (beta ** 2 - 2. * beta) * f0 ** (2. * beta-2)

        xi = nu / alpha * ic
   
        Yxi = tf.where(xi > xip,
                       Yp + (xi - xip)/cap,
                       tf.where(xi < xim,
                                Ym + (xi -xim)/cap,
                                -tf.math.log((tf.math.sqrt(1. + 2. * rho * xi + xi ** 2) - rho - xi) / (1 - rho))
                                )
                       )

        sK0 = tf.where(xi > xip,
                       nu ** 2 / (8. * alpha ** 2 * Yxi) * (-Yp + 3. * (-rho * cap + term) / cap) \
                       + Delta0 / (16. * Yxi) * (2. * cap ** 2 * (Yxi - Yp) + (1 - rho ** 2) * Yp + cap * term - rho),
                       tf.where(xi < xim,
                                nu ** 2 / (8 * alpha ** 2 * Yxi) * (-Ym - 3 * (rho * cap + term) / cap) + Delta0 / (16. * Yxi) * (2 * cap ** 2 * (Yxi - Ym) + (1 - rho ** 2) * Ym - cap * term - rho),
                                nu ** 2 / (8. * alpha ** 2 * Yxi) * (-Yxi + 3 * (xi + rho - rho * tf.math.sqrt(1 + 2 * rho * xi + xi ** 2)) / tf.math.sqrt(1 + 2 * rho * xi + xi ** 2)) \
                                    + Delta0 / (16. * Yxi) * ((1 - rho ** 2) * Yxi + (xi + rho) * tf.math.sqrt(1 + 2 * rho * xi + xi ** 2) - rho)             
                                )
                      )
                    
        theta = -0.5 * rho * nu * alpha * gamma - 2. / 3. * alpha ** 2 * sK0
        highorder = tf.where(theta <0.,
                             tf.math.sqrt(1. - theta * expiries), 
                             1. / tf.math.sqrt(1. + theta * expiries)
                             )
        vol = tf.where(index_atm, volatm, nu * (strikes - forwards) / Yxi * highorder)
        return vol
  