<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.instruments.InterestRateSwap" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="annuity"/>
<meta itemprop="property" content="par_rate"/>
<meta itemprop="property" content="price"/>
</div>

# tf_quant_finance.experimental.instruments.InterestRateSwap

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/instruments/interest_rate_swap.py">View source</a>



Represents a batch of Interest Rate Swaps (IRS).

```python
tf_quant_finance.experimental.instruments.InterestRateSwap(
    start_date, maturity_date, pay_leg, receive_leg, holiday_calendar=None,
    dtype=None, name=None
)
```



<!-- Placeholder for "Used in" -->

An Interest rate swap (IRS) is a contract between two counterparties for an
exchange of a series of payments over a period of time. The payments are made
periodically (for example quarterly or semi-annually) where the last payment
is made at the maturity (or termination) of the contract. In the case of
fixed-for-floating IRS, one counterparty pays a fixed rate while the other
counterparty's payments are linked to a floating index, most commonly the
LIBOR rate. On the other hand, in the case of interest rate basis swap, the
payments of both counterparties are linked to a floating index. Typically, the
floating rate is observed (or fixed) at the begining of each period while the
payments are made at the end of each period [1].

For example, consider a vanilla swap with the starting date T_0 and maturity
date T_n and equally spaced coupon payment dates T_1, T_2, ..., T_n such that

T_0 < T_1 < T_2 < ... < T_n and dt_i = T_(i+1) - T_i    (A)

The floating rate is fixed on T_0, T_1, ..., T_(n-1) and both the fixed and
floating payments are made on T_1, T_2, ..., T_n (payment dates).

The InterestRateSwap class can be used to create and price multiple IRS
simultaneously. The class supports vanilla fixed-for-floating swaps as
well as basis swaps. However all IRS within an IRS object must be priced using
a common reference and discount curve.

#### Example:
The following example illustrates the construction of an IRS instrument and
calculating its price.

```python
import numpy as np
import tensorflow as tf
import tf_quant_finance as tff
dates = tff.experimental.dates
instruments = tff.experimental.instruments
rc = tff.experimental.instruments.rates_common

dtype = np.float64
start_date = dates.convert_to_date_tensor([(2020, 2, 8)])
maturity_date = dates.convert_to_date_tensor([(2022, 2, 8)])
valuation_date = dates.convert_to_date_tensor([(2020, 2, 8)])
period_3m = dates.periods.PeriodTensor(3, dates.PeriodType.MONTH)
period_6m = dates.periods.PeriodTensor(6, dates.PeriodType.MONTH)
fix_spec = instruments.FixedCouponSpecs(
            coupon_frequency=period_6m, currency='usd',
            notional=1., coupon_rate=0.03134,
            daycount_convention=rc.DayCountConvention.ACTUAL_365,
            businessday_rule=dates.BusinessDayConvention.NONE)

flt_spec = instruments.FloatCouponSpecs(
            coupon_frequency=periods_3m, reference_rate_term=periods_3m,
            reset_frequency=periods_3m, currency='usd', notional=1.,
            businessday_rule=dates.BusinessDayConvention.NONE,
            coupon_basis=0., coupon_multiplier=1.,
            daycount_convention=rc.DayCountConvention.ACTUAL_365)

swap = instruments.InterestRateSwap([(2020,2,2)], [(2023,2,2)], [fix_spec],
                                    [flt_spec], dtype=np.float64)

curve_dates = valuation_date + dates.periods.PeriodTensor(
      [1, 2, 3, 5, 7, 10, 30], dates.PeriodType.YEAR)
reference_curve = instruments.RateCurve(
    curve_dates,
    np.array([
      0.02834814, 0.03077457, 0.03113739, 0.03130794, 0.03160892,
      0.03213901, 0.03257991
      ], dtype=dtype),
    dtype=dtype)
market = instruments.InterestRateMarket(
    reference_curve=reference_curve, discount_curve=reference_curve)

price = swap.price(valuation_date, market)
# Expected result: 1e-7
```

#### References:
[1]: Leif B.G. Andersen and Vladimir V. Piterbarg. Interest Rate Modeling,
    Volume I: Foundations and Vanilla Models. Chapter 5. 2010.

#### Args:


* <b>`start_date`</b>: A rank 1 `DateTensor` specifying the dates for the inception
  (start of the accrual) of the swap cpntracts. The shape of the input
  correspond to the numbercof instruments being created.
* <b>`maturity_date`</b>: A rank 1 `DateTensor` specifying the maturity dates for
  each contract. The shape of the input should be the same as that of
  `start_date`.
* <b>`pay_leg`</b>: A list of either `FixedCouponSpecs` or `FloatCouponSpecs`
  specifying the coupon payments for the payment leg of the swap. The
  length of the list should be the same as the number of instruments
  being created.
* <b>`receive_leg`</b>: A list of either `FixedCouponSpecs` or `FloatCouponSpecs`
  specifying the coupon payments for the receiving leg of the swap. The
  length of the list should be the same as the number of instruments
  being created.
* <b>`holiday_calendar`</b>: An instance of <a href="../../../tf_quant_finance/experimental/dates/HolidayCalendar.md"><code>dates.HolidayCalendar</code></a> to specify
  weekends and holidays.
  Default value: None in which case a holiday calendar would be created
  with Saturday and Sunday being the holidays.
* <b>`dtype`</b>: `tf.Dtype`. If supplied the dtype for the real variables or ops
  either supplied to the IRS object or created by the IRS object.
  Default value: None which maps to the default dtype inferred by
  TensorFlow.
* <b>`name`</b>: Python str. The name to give to the ops created by this class.
  Default value: `None` which maps to 'interest_rate_swap'.

## Methods

<h3 id="annuity"><code>annuity</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/instruments/interest_rate_swap.py">View source</a>

```python
annuity(
    valuation_date, market, model=None
)
```

Returns the annuity of each swap on the vauation date.


<h3 id="par_rate"><code>par_rate</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/instruments/interest_rate_swap.py">View source</a>

```python
par_rate(
    valuation_date, market, model=None
)
```

Returns the par swap rate for the swap.


<h3 id="price"><code>price</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/instruments/interest_rate_swap.py">View source</a>

```python
price(
    valuation_date, market, model=None
)
```

Returns the present value of the instrument on the valuation date.


#### Args:


* <b>`valuation_date`</b>: A scalar `DateTensor` specifying the date on which
  valuation is being desired.
* <b>`market`</b>: A namedtuple of type `InterestRateMarket` which contains the
  necessary information for pricing the interest rate swap.
* <b>`model`</b>: Reserved for future use.


#### Returns:

A Rank 1 `Tensor` of real type containing the modeled price of each IRS
contract based on the input market data.




