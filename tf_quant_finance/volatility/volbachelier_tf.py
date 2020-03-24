"""
Created on Fri Nov 22 15:22:13 2019

# Copyright 2020 Joerg Kienitz

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

@author: Joerg Kienitz
"""


 
import math as m
import tensorflow.compat.v2 as tf

def func1(t,
          x,
          z,
          betaStart,
          betaEnd,
          cut_off2,
          cut_off3,
          bLFK4,
          cLFK4,
          dLFK4):
    """ helper function
    
    """
    
    u = -(tf.math.log(z) + betaStart) / (betaEnd - betaStart)
    
    num = tf.where(u < cut_off2,
                   bLFK4[0] + u * (bLFK4[1] +u * (bLFK4[2] +u * (bLFK4[3] + u * (bLFK4[4] +u * (bLFK4[5] +u * (bLFK4[6] + u * (bLFK4[7] +u * (bLFK4[8] +u * bLFK4[9] )))) )))), 
                   tf.where(u < cut_off3, 
                            cLFK4[0] + u * (cLFK4[1] +u * (cLFK4[2] +u * (cLFK4[3] +u * (cLFK4[4] +u * (cLFK4[5] +u * (cLFK4[6] + u * (cLFK4[7] + u * (cLFK4[8] + u * cLFK4[9] )))))))),
                            dLFK4[0] + u * (dLFK4[1] +u * (dLFK4[2] +u * (dLFK4[3] +u * (dLFK4[4] +u * (dLFK4[5] +u * (dLFK4[6] + u * (dLFK4[7] + u * (dLFK4[8] + u * dLFK4[9] )))))))) 
                            )
                   )
    
    den = tf.where(u < cut_off2,
                   1.0 + u * (bLFK4[10] + u * (bLFK4[11] + u * (bLFK4[12] + u * (bLFK4[13] + u * (bLFK4[14] + u * (bLFK4[15] + u * (bLFK4[16] ) ))))) ),
                   tf.where(u < cut_off3, 
                            1.0 + u * (cLFK4[10] + u * (cLFK4[11] + u * (cLFK4[12] + u * (cLFK4[13] + u * (cLFK4[14] + u * (cLFK4[15] + u * (cLFK4[16]))))))),
                            1.0 + u * (dLFK4[10] + u * (dLFK4[11] + u * (dLFK4[12] + u * (dLFK4[13] + u * (dLFK4[14] + u * (dLFK4[15] + u * (dLFK4[16])))))))
                            )
                   )    
    hz = num / den 
    return tf.math.abs(x) / (tf.math.sqrt(hz * t ) )

def func2(price,
          t,
          x,
          aLFK4):
    """ helper function
    
    """
    p = tf.where(x < 0., price - x, price)
    u = eta(tf.math.abs(x) / p )
    num = aLFK4[0] + u * (aLFK4[1] + u * (aLFK4[2] + u * (aLFK4[3] + u * (aLFK4[4] + u * (aLFK4[5] + u * (aLFK4[6] + u * (aLFK4[7])))))))
    den = 1.0 + u * (aLFK4[8] + u * (aLFK4[9] + u * (aLFK4[10] + u * (aLFK4[11] + u * (aLFK4[12]))))) 
    return p * num / den / tf.math.sqrt(t)

def eta(z):
# case for avoiding incidents of 0/0, z close to zero
    return tf.where(z < 1e-2, 
        1 -z *(0.5+ z * (1.0 / 12.0 + z * (1.0 / 24.0 + z * (19.0 / 720.0 + z * (3.0 / 160.0 + z * (863.0/60480.0 + z * (275.0/24192.0))))))),
        -z / tf.math.log1p(-z))       

def vol_atm(price, t):
    # atm case
    return price * tf.math.sqrt(2. * m.pi / t)

def vol_iotm(x,betaStart,betaEnd,price, t, cut_off1,cut_off2,cut_off3, aLFK4,bLFK4,cLFK4,dLFK4):
    # other cases (ITM/OTM)
    z = tf.where(x>=0.,(price - x) / x, -price / x)       
        
    return tf.where(z <= cut_off1, 
                   func1(t,x,z,betaStart,betaEnd,cut_off2,cut_off3,bLFK4,cLFK4,dLFK4), 
                   func2(price,t,x,aLFK4)
                   )
 


    
def volbachelier_tf(signs, 
                    strikes, 
                    forwards, 
                    expiries, 
                    prices):
    """ computes the Bachelier implied volatility for a batch of prices of 
    European Call or Put options.
    We assume a standard Brownian motion of the form
       dS = sigma dW
    for the underlying. sigma is the implied volatility.

   sign : A `Tensor` of any shape. The current sign that specifies if the 
         prices are Call (sign=1) or Put (sign=-1).
    strikes : A real `Tensor`. The strikes of the options to be priced.
    forwards : A real `Tensor` of same shape and dtype as `strikes`.
    expiries : A real `Tensor` of same shape and dtype as `strikes`. 
    prices : A real `Tensor` of same shape and dtype as `strikes`. The prices
        of the European Call, resp. Put prices

    Returns

    implied_bachelier_vols: A `Tensor` of the same shape as `strieks`. 
    The Bachelier implied volatilties of the Call, resp. Put options.
    

References:
   [1] Fabien Le Floc'h, "Fast and Accurate Analytic Basis Point Volatility"
   Link: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2420757


"""
    sign = tf.convert_to_tensor(signs, dtype=tf.float64, name='sign')
    strike = tf.convert_to_tensor(strikes, dtype=tf.float64, name='strikes')
    forward = tf.convert_to_tensor(forwards, dtype=tf.float64, name='forward')
    t = tf.convert_to_tensor(expiries, dtype=tf.float64, name='expiries')
    price = tf.convert_to_tensor(prices, dtype=tf.float64, name='price')

    # for rational expansion
    aLFK4 = tf.constant([0.06155371425063157,2.723711658728403,10.83806891491789,
             301.0827907126612,1082.864564205999, 790.7079667603721, 
             109.330638190985, 0.1515726686825187, 1.436062756519326, 
             118.6674859663193, 441.1914221318738, 313.4771127147156, 
             40.90187645954703], dtype=tf.float64)
    bLFK4 = tf.constant([0.6409168551974357, 788.5769356915809, 445231.8217873989, 
             149904950.4316367, 32696572166.83277, 4679633190389.852, 
             420159669603232.9, 2.053009222143781e+16, 3.434507977627372e+17, 
             2.012931197707014e+16, 644.3895239520736, 211503.4461395385, 
             42017301.42101825, 5311468782.258145, 411727826816.0715, 
             17013504968737.03, 247411313213747.3], dtype=tf.float64)
    cLFK4 = tf.constant([0.6421106629595358, 654.5620600001645, 291531.4455893533, 
             69009535.38571493, 9248876215.120627, 
             479057753706.175, 9209341680288.471, 61502442378981.76, 
             107544991866857.5, 63146430757.94501, 
             437.9924136164148, 90735.89146171122, 9217405.224889684, 
             400973228.1961834, 7020390994.356452, 
             44654661587.93606, 76248508709.85633], dtype=tf.float64)
    dLFK4 = tf.constant([0.936024443848096, 328.5399326371301, 177612.3643595535, 
                         8192571.038267588, 110475347.0617102, 
             545792367.0681282, 1033254933.287134, 695066365.5403566, 
             123629089.1036043, 756.3653755877336, 
             173.9755977685531, 6591.71234898389, 82796.56941455391, 
             396398.9698566103, 739196.7396982114, 
             493626.035952601, 87510.31231623856], dtype=tf.float64)
    
    #eps = tf.cast(tf.constant(np.finfo(float).tiny),tf.float32)
    # constants
    cut_off_atm = tf.constant(1.0e-10, dtype=tf.float64) # ATM Cutoff level #np.finfo(float).eps
    cut_off1 = tf.constant(0.15, dtype=tf.float64)       # cut-off for -C(x)/x
    cut_off2 = tf.constant(0.0091, dtype=tf.float64)     # 1st cut-off for tilde(eta)
    cut_off3 = tf.constant(0.088, dtype=tf.float64)      # 2nd cut-off for tilde(eta)
    
    betaStart = - tf.math.log(cut_off1) ; betaEnd = 708.3964185322641#- tf.math.log(machine eps)    
    
    x = ( forward - strike) * sign   # intrinsic value of the Call (sign=1) 
                                     # or Put (sign = -1)
    implied_bachelier_vols = tf.where(tf.math.abs(x) < cut_off_atm, 
                   vol_atm(price,t), 
                   vol_iotm(x,betaStart,betaEnd,price, t, cut_off1,cut_off2,cut_off3, aLFK4,bLFK4,cLFK4,dLFK4)
                  )
    
    # return the full tensor of implied Bachelier volatilities due to atm and itm/otm         
    return implied_bachelier_vols 


