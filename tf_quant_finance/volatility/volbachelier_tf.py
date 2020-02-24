# -*- coding: utf-8 -*-
"""

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



    # aLFK4 = [0.06155371425063157, 2.723711658728403, 10.83806891491789, 301.0827907126612 , 1082.864564205999 , 790.7079667603721 , 109.330638190985 , 0.1515726686825187 , 1.436062756519326 , 118.6674859663193 , 441.1914221318738 , 313.4771127147156 , 40.90187645954703];
    # c0=0.6409168551974356; 
    # c1=776.7622553541449; d1=640.570978803313;
    # c2=431496.7672664836; d2=206873.1616020722;
    # c3=142810081.2530825; d3=40411807.74439474;
    # c4=30593703611.75923; d4=5007804896.265911;
    # c5=4296256150040.825; d5=379858284395.2218;
    # c6=377808909050483.9; d6=15253797078346.91;
    # c7=1.799539603508817e+16; d7=211469320780659.9;
    # c8=2.864267851212242e+17;
    # c9=1.505975341130321e+16;

    # e0=0.6421698396894946; 
    # e1=639.0799338046976; f1=428.4860093838116;
    # e2=278070.4504753253; f2=86806.89002606465;
    # e3=64309618.34521588; f3=8635134.393384729;
    # e4=8434470508.516712; f4=368872214.1525768;
    # e5=429163238246.6056; f5=6359299149.626331;
    # e6=8127970878235.127; f6=39926015967.88848;
    # e7=53601225394979.81; f7=67434966969.06365;
    # e8=92738918006503.35;
    # e9=54928597545.97237;

    # g0=0.9419766804760195; 
    # g1=319.5904313022832; h1=170.3825619167351;
    # g2=169280.1584005307; h2=6344.159541465554;
    # g3=7680298.116948191; h3=78484.04408022196;
    # g4=102052455.1237945; h4=370696.1131305614;
    # g5=497528976.6077898; h5=682908.5433659635;
    # g6=930641173.0039455; h6=451067.0450625782;
    # g7=619268950.1849232; h7=79179.06152239779;
    # g8=109068992.0230439;
    # g9=672.856898188759;
    
    # bLFK4 = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, d1, d2, d3, d4, d5, d6, d7]
    # cLFK4 = [e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, f1, f2, f3, f4, f5, f6, f7]
    # dLFK4 = [g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, h1, h2, h3, h4, h5, h6, h7]


"""
import math as m
import tensorflow as tf

def func1(t,x,z,betaStart,betaEnd,cut_off2,cut_off3,bLFK4,cLFK4,dLFK4):
    u = -(tf.math.log(z)+ betaStart) /(betaEnd - betaStart)
    
    num = tf.where(u < cut_off2,
                   bLFK4[0] + u *(bLFK4[1] +u *(bLFK4[2] +u *(bLFK4[3] + u *(bLFK4[4] +u *(bLFK4[5] +u *(bLFK4[6] + u *(bLFK4[7] +u *(bLFK4[8] +u * bLFK4[9] )))) )))), 
                   tf.where(u < cut_off3, 
                            cLFK4[0] + u *(cLFK4[1] +u *(cLFK4[2] +u *(cLFK4[3] +u *(cLFK4[4] +u *(cLFK4[5] +u *(cLFK4[6] + u *(cLFK4[7] +u *(cLFK4[8] +u * cLFK4[9] )))) )))),
                            dLFK4[0] + u *(dLFK4[1] +u *(dLFK4[2] +u *(dLFK4[3] +u *( dLFK4[4] +u *( dLFK4[5] +u *( dLFK4[6] + u *( dLFK4[7] +u *(dLFK4[8]+u * dLFK4[9] )))) )))) 
                            )
                   )
    
    den = tf.where(u < cut_off2,
                   1.0 + u *(bLFK4[10] +u *(bLFK4[11] + u *(bLFK4[12] +u *(bLFK4[13] +u *(bLFK4[14] +u *(bLFK4[15] +u *(bLFK4[16] ) ))))) ),
                   tf.where(u < cut_off3, 
                            1.0 + u *(cLFK4[10] +u *(cLFK4[11] + u *(cLFK4[12] +u *(cLFK4[13] +u *(cLFK4[14] +u *(cLFK4[15] +u *(cLFK4[16]))))))),
                            1.0 + u *(dLFK4[10] +u *(dLFK4[11] + u *(dLFK4[12] +u *( dLFK4[13] +u *( dLFK4[14] +u *( dLFK4[15] +u *( dLFK4[16] ) ))))) )
                            )
                   )    
    hz = num / den 
    return tf.math.abs(x) / (tf.math.sqrt(hz * t ) )

def func2(price,t,x,aLFK4):
    p = tf.where(x < 0., price - x, price)
    u = eta(tf.math.abs(x) / p )
    num = aLFK4[0] + u *( aLFK4[1] +u *( aLFK4[2] +u *( aLFK4[3] +u *( aLFK4[4] + u *( aLFK4[5] +u *( aLFK4[6] + u *( aLFK4[7]))) ))))
    den = 1.0 + u *( aLFK4[8] +u *( aLFK4[9] + u *( aLFK4[10] +u *( aLFK4[11] +u *( aLFK4[12] ))))) 
    return p * num / den / tf.math.sqrt(t)

def eta(z):
# case for avoiding incidents of 0/0, z close to zero...
    return tf.where(z < 1e-2, 
        1 -z *(0.5+ z *(1.0/12+ z *(1.0/24+ z *(19.0/720+ z *(3.0/160+ z *(863.0/60480+ z *(275.0/24192) )))))),
        -z/ tf.math.log1p(-z) )       

def vol_atm(price, t):
    # atm case
    return price * tf.math.sqrt(2. * m.pi / t)

def vol_iotm(x,betaStart,betaEnd,price, t, cut_off1,cut_off2,cut_off3, aLFK4,bLFK4,cLFK4,dLFK4):
    # other cases (ITM/OTM)
    z = tf.where(x>=0.,(price-x)/x, -price/x)       
        
    return tf.where(z <= cut_off1, 
                   func1(t,x,z,betaStart,betaEnd,cut_off2,cut_off3,bLFK4,cLFK4,dLFK4), 
                   func2(price,t,x,aLFK4)
                   )
 


    
def volbachelier_tf(sign, strike, forward, t, price):
    sign = tf.convert_to_tensor(sign, dtype=tf.float64, name='forwards')
    strike = tf.convert_to_tensor(strike, dtype=tf.float64, name='strikes')
    forward = tf.convert_to_tensor(forward, dtype=tf.float64, name='expiries')
    t = tf.convert_to_tensor(t, dtype=tf.float64, name='displacement')
    price = tf.convert_to_tensor(price, dtype=tf.float64, name='nu')

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
    cLFK4 = tf.constant([0.6421106629595358, 654.5620600001645, 291531.4455893533, 69009535.38571493, 9248876215.120627, 
             479057753706.175, 9209341680288.471, 61502442378981.76, 107544991866857.5, 63146430757.94501, 
             437.9924136164148, 90735.89146171122, 9217405.224889684, 400973228.1961834, 7020390994.356452, 
             44654661587.93606, 76248508709.85633], dtype=tf.float64)
    dLFK4 = tf.constant([0.936024443848096, 328.5399326371301, 177612.3643595535, 8192571.038267588, 110475347.0617102, 
             545792367.0681282, 1033254933.287134, 695066365.5403566, 123629089.1036043, 756.3653755877336, 
             173.9755977685531, 6591.71234898389, 82796.56941455391, 396398.9698566103, 739196.7396982114, 
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

            
    return tf.where(tf.math.abs(x) < cut_off_atm, 
                   vol_atm(price,t), 
                   vol_iotm(x,betaStart,betaEnd,price, t, cut_off1,cut_off2,cut_off3, aLFK4,bLFK4,cLFK4,dLFK4)
                  )    


