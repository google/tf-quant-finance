from tf_quant_finance.volatility import barrier_option
import tensorflow as tf


class BarrierOptionTest(tf.test.TestCase):  
    def test_price_barrier_option_1d(self):
        asset_price = 100.0
        rebate = 3.0
        time_to_maturity = 0.5
        rate = 0.08
        b = 0.04
        asset_yield = -(b-rate)
        strike_price, barrier_price, cp, ba, price_true, mp = self.get_vals("pdi")
        volitility = 0.25
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
        price = barrier_option.price_barrier_option(cp, ba, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity, mp)
        self.assertAllClose(price, price_true, 10e-3)

    def get_vals(self, param):
        if param == "cdo":
            return 90, 95, 1., 1., 9.0246, 5
        elif param == "cdi":
            return 90, 95, 1., 1., 7.7627, 1
        elif param == "cuo":
            return 90, 105, 1., -1., 2.6789, 7
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
        
    def test_price_barrier_option_2d(self):
        asset_price = [100., 100., 100., 100., 100., 100., 100., 100.]
        rebate = [3., 3., 3., 3., 3., 3., 3., 3.]
        time_to_maturity = [.5, .5, .5, .5, .5, .5, .5, .5]
        rate = [.08, .08, .08, .08, .08, .08, .08, .08]
        volitility = [.25, .25, .25, .25, .25, .25, .25, .25]
        strike_price = [90., 90., 90., 90., 90., 90., 90., 90.]
        barrier_price = [95., 95., 105., 105., 95., 105., 95., 105.]
        cp = [1., 1., 1., 1., -1., -1., -1., -1.]
        ba = [1., 1., -1., -1., 1., -1., 1., -1.]
        price_true = [9.024, 7.7627, 2.6789, 14.1112, 2.2798, 3.7760, 2.95586, 1.4653]
        mp = [5, 1, 7 ,3 ,6 ,8, 2, 4]
        asset_yield = [.04, .04, .04, .04, .04, .04, .04, .04]
        price = barrier_option.price_barrier_option(cp, ba, rate, asset_yield, asset_price, strike_price, barrier_price, rebate, volitility, time_to_maturity, mp)
        self.assertAllClose(price, price_true, 10e-3)

if __name__ == '__main__':
    tf.test.main()
