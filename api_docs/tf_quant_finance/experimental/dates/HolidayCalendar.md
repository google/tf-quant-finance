<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates.HolidayCalendar" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="add_business_days"/>
<meta itemprop="property" content="add_period_and_roll"/>
<meta itemprop="property" content="business_days_between"/>
<meta itemprop="property" content="business_days_in_period"/>
<meta itemprop="property" content="is_business_day"/>
<meta itemprop="property" content="roll_to_business_day"/>
<meta itemprop="property" content="subtract_business_days"/>
<meta itemprop="property" content="subtract_period_and_roll"/>
</div>

# tf_quant_finance.experimental.dates.HolidayCalendar

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>



Represents a holiday calendar.

```python
tf_quant_finance.experimental.dates.HolidayCalendar(
    weekend_mask=None, holidays=None, start_year=None, end_year=None
)
```



<!-- Placeholder for "Used in" -->

Provides methods for manipulating the dates taking into account the holidays,
and the business day roll conventions. Weekends are treated as holidays.

#### Args:


* <b>`weekend_mask`</b>: Sequence of 7 elements, where "0" means work day and "1" -
  day off. The first element is Monday. By default, no weekends are
  applied. Some of the common weekend patterns are defined in
  <a href="../../../tf_quant_finance/experimental/dates/WeekendMask.md"><code>dates.WeekendMask</code></a>.
  Default value: None which maps to no weekend days.
* <b>`holidays`</b>: Defines the holidays that are added to the weekends defined by
  `weekend_mask`. Can be provided in following forms:
  - Iterable of tuples containing dates in (year, month, day) format:
    ```python
    holidays = [(2020, 1, 1), (2020, 12, 25),
                (2021, 1, 1), (2021, 12, 24)]
    ```
  - Iterable of datetime.date objects:
    ```python
    holidays = [datetime.date(2020, 1, 1), datetime.date(2020, 12, 25),
                datetime.date(2021, 1, 1), datetime.date(2021, 12, 24)]
    ```
  - A numpy array of type np.datetime64:

    ```python
    holidays = np.array(['2020-01-01', '2020-12-25', '2021-01-01',
                         '2020-12-24'], dtype=np.datetime64)
    ```

  Note that it is necessary to provide holidays for each year, and also
  adjust the holidays that fall on the weekends if required, like
  2021-12-25 to 2021-12-24 in the example above. To avoid doing this
  manually one can use AbstractHolidayCalendar from Pandas:

  ```python
  from pandas.tseries.holiday import AbstractHolidayCalendar
  from pandas.tseries.holiday import Holiday
  from pandas.tseries.holiday import nearest_workday

  class MyCalendar(AbstractHolidayCalendar):
      rules = [
          Holiday('NewYear', month=1, day=1, observance=nearest_workday),
          Holiday('Christmas', month=12, day=25,
                   observance=nearest_workday)
      ]

  calendar = MyCalendar()
  holidays_index = holidays.holidays(
      start=datetime.date(2020, 1, 1),
      end=datetime.date(2030, 12, 31))
  holidays = np.array(holidays_index.to_pydatetime(), dtype="<M8[D]")
  ```
* <b>`start_year`</b>: Integer giving the earliest year this calendar includes. If
  `holidays` is specified, then `start_year` and `end_year` are ignored,
  and the boundaries are derived from `holidays`. If `holidays` is `None`,
  both `start_year` and `end_year` must be specified.
* <b>`end_year`</b>: Integer giving the latest year this calendar includes. If
  `holidays` is specified, then `start_year` and `end_year` are ignored,
  and the boundaries are derived from `holidays`. If `holidays` is `None`,
  both `start_year` and `end_year` must be specified.

## Methods

<h3 id="add_business_days"><code>add_business_days</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
add_business_days(
    date_tensor, num_days,
    roll_convention=tf_quant_finance.experimental.dates.BusinessDayConvention.NONE
)
```

Adds given number of business days to given dates.

Note that this is different from calling `add_period_and_roll` with
PeriodType.DAY. For example, adding 5 business days to Monday gives the next
Monday (unless there are holidays on this week or next Monday). Adding 5
days and rolling means landing on Saturday and then rolling either to next
Monday or to Friday of the same week, depending on the roll convention.

If any of the dates in `date_tensor` are not business days, they will be
rolled to business days before doing the addition. If `roll_convention` is
`NONE`, and any dates are not business days, an exception is raised.

#### Args:


* <b>`date_tensor`</b>: DateTensor of dates to advance from.
* <b>`num_days`</b>: Tensor of int32 type broadcastable to `date_tensor`.
* <b>`roll_convention`</b>: BusinessDayConvention. Determines how to roll a date that
  falls on a holiday.


#### Returns:

The resulting DateTensor.


<h3 id="add_period_and_roll"><code>add_period_and_roll</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
add_period_and_roll(
    date_tensor, period_tensor,
    roll_convention=tf_quant_finance.experimental.dates.BusinessDayConvention.NONE
)
```

Adds given periods to given dates and rolls to business days.

The original dates are not rolled prior to addition.

#### Args:


* <b>`date_tensor`</b>: DateTensor of dates to add to.
* <b>`period_tensor`</b>: PeriodTensor broadcastable to `date_tensor`.
* <b>`roll_convention`</b>: BusinessDayConvention. Determines how to roll a date that
  falls on a holiday.


#### Returns:

The resulting DateTensor.


<h3 id="business_days_between"><code>business_days_between</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
business_days_between(
    from_dates, to_dates
)
```

Calculates number of business between pairs of dates.

For each pair, the initial date is included in the difference, and the final
date is excluded. If the final date is the same or earlier than the initial
date, zero is returned.

#### Args:


* <b>`from_dates`</b>: DateTensor of initial dates.
* <b>`to_dates`</b>: DateTensor of final dates, should be broadcastable to
  `from_dates`.


#### Returns:

An int32 Tensor with the number of business days between the
corresponding pairs of dates.


<h3 id="business_days_in_period"><code>business_days_in_period</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
business_days_in_period(
    date_tensor, period_tensor
)
```

Calculates number of business days in a period.

Includes the dates in `date_tensor`, but excludes final dates resulting from
addition of `period_tensor`.

#### Args:


* <b>`date_tensor`</b>: DateTensor of starting dates.
* <b>`period_tensor`</b>: PeriodTensor, should be broadcastable to `date_tensor`.


#### Returns:

An int32 Tensor with the number of business days in given periods that
start at given dates.


<h3 id="is_business_day"><code>is_business_day</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
is_business_day(
    date_tensor
)
```

Returns a tensor of bools for whether given dates are business days.


<h3 id="roll_to_business_day"><code>roll_to_business_day</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
roll_to_business_day(
    date_tensor, roll_convention
)
```

Rolls the given dates to business dates according to given convention.


#### Args:


* <b>`date_tensor`</b>: DateTensor of dates to roll from.
* <b>`roll_convention`</b>: BusinessDayConvention. Determines how to roll a date that
  falls on a holiday.


#### Returns:

The resulting DateTensor.


<h3 id="subtract_business_days"><code>subtract_business_days</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
subtract_business_days(
    date_tensor, num_days,
    roll_convention=tf_quant_finance.experimental.dates.BusinessDayConvention.NONE
)
```

Adds given number of business days to given dates.

Note that this is different from calling `subtract_period_and_roll` with
PeriodType.DAY. For example, subtracting 5 business days from Friday gives
the previous Friday (unless there are holidays on this week or previous
Friday). Subtracting 5 days and rolling means landing on Sunday and then
rolling either to Monday or to Friday, depending on the roll convention.

If any of the dates in `date_tensor` are not business days, they will be
rolled to business days before doing the subtraction. If `roll_convention`
is `NONE`, and any dates are not business days, an exception is raised.

#### Args:


* <b>`date_tensor`</b>: DateTensor of dates to advance from.
* <b>`num_days`</b>: Tensor of int32 type broadcastable to `date_tensor`.
* <b>`roll_convention`</b>: BusinessDayConvention. Determines how to roll a date that
  falls on a holiday.


#### Returns:

The resulting DateTensor.


<h3 id="subtract_period_and_roll"><code>subtract_period_and_roll</code></h3>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/holiday_calendar.py">View source</a>

```python
subtract_period_and_roll(
    date_tensor, period_tensor,
    roll_convention=tf_quant_finance.experimental.dates.BusinessDayConvention.NONE
)
```

Subtracts given periods from given dates and rolls to business days.

The original dates are not rolled prior to subtraction.

#### Args:


* <b>`date_tensor`</b>: DateTensor of dates to subtract from.
* <b>`period_tensor`</b>: PeriodTensor broadcastable to `date_tensor`.
* <b>`roll_convention`</b>: BusinessDayConvention. Determines how to roll a date that
  falls on a holiday.


#### Returns:

The resulting DateTensor.




