from tf_quant_finance.volatility import barrier_option
import tensorflow as tf
from scipy.stats import norm
from math import log, exp


class BarrierOptionTest(tf.test.TestCase):
    def _calc_real_params(self, rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity):
        """ Function calculates params to test against the construct_params method """
        time_volitility_real = volitility*time_to_maturity**.5
        mu = (rate-asset_yield)-((volitility**2)/2)
        lamda_real = 1+(mu/(volitility**2))
        b_real = ((mu**2+(2*volitility**2*rate))**.5)/(volitility**2)
        a_real = mu/(volitility**2)
        t_voli = volitility*(time_to_maturity**.5)
        x_real = (log(asset_price/strike_price)/(t_voli))+(lamda_real*t_voli)
        x1_real = (log(asset_price/barrier_price)/(t_voli))+(lamda_real*t_voli)
        y_real = (log(barrier_price**2/(asset_price*strike_price))/(t_voli))+(lamda_real*t_voli)
        y1_real = (log(barrier_price/asset_price)/(t_voli))+(lamda_real*t_voli)
        z_real = (log(barrier_price/asset_price)/(t_voli))+(b_real*t_voli)
        return x_real, x1_real, y_real, y1_real, lamda_real, z_real, a_real, b_real, time_volitility_real

    def _i12(self, call_or_put, rate, asset_price, asset_yield, time_to_maturity, strike_price, params, val):
        x = params[val]
        time_volitility = params["time_volitility"]
        cdf1 = norm.cdf(call_or_put*x)
        term1 = call_or_put*asset_price*exp(-1*asset_yield*time_to_maturity)*cdf1
        cdf2 = norm.cdf((call_or_put*x)-(call_or_put*time_volitility))
        term2 = call_or_put*strike_price*exp(-1*rate*time_to_maturity)*cdf2
        return term1-term2

    def _i34(self, call_or_put, below_or_above, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, val):
        y = params[val]
        time_volitility = params["time_volitility"]
        lamda = params["lambda"]
        cdf1 = norm.cdf(below_or_above*y)
        term1 = call_or_put*asset_price*exp(-1*asset_yield*time_to_maturity)*((barrier_price/asset_price)**(2*lamda))*cdf1
        cdf2 = norm.cdf((below_or_above*y)-(below_or_above*time_volitility))
        term2 = call_or_put*strike_price*exp(-1*rate*time_to_maturity)*((barrier_price/asset_price)**((2*lamda)-2))*cdf2
        return term1-term2

    def _i5(self, below_or_above, pay_off, rate, asset_price, time_to_maturity, barrier_price, params):
        x1 = params["x1"]
        y1 = params["y1"]
        lamda = params["lambda"]
        time_volitility = params["time_volitility"]
        div = barrier_price/asset_price
        cdf1 = norm.cdf(below_or_above*x1-below_or_above*time_volitility)
        cdf2 = norm.cdf(below_or_above*y1-below_or_above*time_volitility)
        diff = cdf1-((div**(2*lamda-2))*cdf2)
        ex = exp(-1*rate*time_to_maturity)
        return pay_off*ex*diff

    def _i6(self, below_or_above, pay_off, asset_price, barrier_price, params):
        a = params["a"]
        b = params["b"]
        z = params["z"]
        time_volitility = params["time_volitility"]
        div = barrier_price/asset_price
        cdf1 = norm.cdf(below_or_above*z)
        cdf2 = norm.cdf(below_or_above*z-2.*below_or_above*b*time_volitility)
        div1 = div**(a+b)
        div2 = div**(a-b)
        diff = div1*cdf1+div2*cdf2
        return pay_off*diff
    
    def test_construct_params_1d(self):
        """Function tests construct_params when 1d is passed"""
        params = barrier_option._construct_params(0.2, 0.1, 150.0, 100.0, 50.0, 1.0, 4.0)
        real_val = self._calc_real_params(0.2, 0.1, 150.0, 100.0, 50.0, 1.0, 4.0)
        # [1.4027, 1.7493, 0.3041, 0.6507, 0.6, 0.9474, -0.4, 0.7483, 2.0]
        self.assertAllClose(params, real_val)


    def test_construct_params_2d(self):
        """Function tests construct_params when two dim values passed"""
        # each element in vector is for a different security
        rate = tf.convert_to_tensor([0.2, 0.1])
        asset_yield = tf.convert_to_tensor([0.1, 0.2])
        asset_price = tf.convert_to_tensor([150.0, 100.0])
        strike_price = tf.convert_to_tensor([100.0, 50.0])
        barrier_price = tf.convert_to_tensor([50.0, 25.0])
        volitility = tf.convert_to_tensor([1.0, 0.5])
        time_to_maturity = tf.convert_to_tensor([4.0,2.0])
        real_vals1 = self._calc_real_params(.2, .1, 150.0, 100.0, 50.0, 1.0, 4.0)
        real_vals2 = self._calc_real_params(.1, .2, 100.0, 50.0, 25.0, 0.5, 2.0)
        rtn_vals = barrier_option._construct_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity)
        real_vals = [[i,j] for i,j in zip(real_vals1,real_vals2)]
        self.assertAllClose(real_vals, rtn_vals)

  
    def test_price_barrier_option_1d(self):
        asset_price = 100.0
        rebate = 3.0
        time_to_maturity = 0.5
        rate = 0.08
        b = 0.04
        asset_yield = -(b-rate)
        strike_price, barrier_price, cp, ba, price_true, mp = self.get_vals("cdo")
        volitility = 0.25
        param_vals = self._calc_real_params(rate, asset_yield, asset_price, strike_price, barrier_price, volitility, time_to_maturity)
        params = {
            "x": param_vals[0],
            "x1": param_vals[1],
            "y": param_vals[2],
            "y1": param_vals[3],
            "lambda": param_vals[4],
            "z": param_vals[5],
            "a": param_vals[6],
            "b": param_vals[7],
            "time_volitility": param_vals[8]
        }
        """
        1 -> cdi
        2 -> pdi
        3 -> cui
        4 -> pui
        5 -> cdo //fails I6 is calculated incorrectly
        6 -> pdo //fails
        7 -> cuo //fails
        8 -> puo //fails
        """
        price = barrier_option.price_barrier_option([cp], [ba], rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity, mp)
        print("I1#############################: ", self._i12(cp, rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x"))
        print("I2#############################: ", self._i12(cp, rate, asset_price, asset_yield, time_to_maturity, strike_price, params, "x1"))
        print("I3#############################: ", self._i34(cp, ba, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y"))
        print("I4#############################: ", self._i34(cp, ba, rate, asset_price, asset_yield, time_to_maturity, strike_price, barrier_price, params, "y1"))
        print("I5#############################: ", self._i5(ba, rebate, rate, asset_price, time_to_maturity, barrier_price, params))
        print("I6#############################: ", self._i6(ba, rebate, asset_price, barrier_price, params))
        print("price: ", price)
        print("True: ", price_true)

    def get_vals(self, param):
        if param == "cdo":
            return 90, 95, 1., 1., 9.0246, 5
        elif param == "cdi":
            return 90, 95, 1., 1., 7.7627, 1
        elif param == "cuo":
            return 90, 105, 1., -1., 2.3580, 7
        elif param == "cui":
            return 90, 105, 1., -1., 14.1112, 3
        elif param == "pdo":
            return 90, 95, -1., 1., 2.2798, 6
        elif param == "puo":
            return 90, 105, -1., -1., 3.7760, 8
        elif param == "pdi":
            return 90, 95, -1., 1., 2.9586, 2
        elif param == "pui":
            return 90, 105, -1., -1., 1.4653, 4
        
    def price_barrier_option_2d(self):
        asset_price = [100.0,100.0]
        rebate = [3.0,3.0]
        time_to_maturity = [0.5,0.5]
        rate = [0.08,0.08]
        strike_price = [100.0,100.0]
        barrier_price = [95.0,95.0]
        volitility = [0.25,0.25]
        b = 0.04
        asset_yield = [-(b-rate[0]), -(b-rate[0])]
        price = barrier_option.price_barrier_option([1.0,1.0], rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity, 1)
        print("price: ", price)
        print("\nTrue: ", 4.0109)

if __name__ == '__main__':
    tf.test.main()


"""
Down and In call
strike_price > barrier_price [0, 0, 1, 0, 1, 0] or strike_price < barrier_price [1, -1, 0, 1, 1, 0] [1,1]
Down and in put
strike_price > barrier_price [0, 1, -1, 1, 1, 0] or strike_price < barrier_price [1, 0, 0, 0, 1, 0] [1,-1]
Up and in call
[1, 0, 0, 0, 1, 0] or [0, 1, -1, 1, 1, 0] [-1,1]
Up and in put
[1, -1, 0, 1, 1, 0] or [0, 0, 1, 0, 1, 0] [-1,-1]

#if otype == "out":
# Out options become worthless if asset price S hit the barrier before expiration
# Down and out call S > H
# Down and out put I6
# Up and out call S < H I6
# Up and out put
#   pass
# elif otype == "in":
# In options become in play if asset price S hits Barrier H before expiration
# Down and in call S > H 
            # Up and in call S < H
# Down and in put S > H
# Up and in put S < H
#    pass
# return

"""
