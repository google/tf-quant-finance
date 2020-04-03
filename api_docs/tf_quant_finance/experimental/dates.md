<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf_quant_finance.experimental.dates" />
<meta itemprop="path" content="Stable" />
</div>

# Module: tf_quant_finance.experimental.dates

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">
</table>

<a target="_blank" href="https://github.com/google/tf-quant-finance/blob/master/tf_quant_finance/experimental/dates/__init__.py">View source</a>



Date-related utilities.



## Modules

[`date_utils`](../../tf_quant_finance/experimental/dates/date_utils.md) module: Utilities for working with dates.

[`daycounts`](../../tf_quant_finance/experimental/dates/daycounts.md) module: Day Count Conventions.

[`periods`](../../tf_quant_finance/experimental/dates/periods.md) module: PeriodTensor definition.

## Classes

[`class BusinessDayConvention`](../../tf_quant_finance/experimental/dates/BusinessDayConvention.md): Conventions that determine how to roll dates that fall on holidays.

[`class BusinessDaySchedule`](../../tf_quant_finance/experimental/dates/BusinessDaySchedule.md): Generates schedules containing every business day in a period.

[`class DateTensor`](../../tf_quant_finance/experimental/dates/DateTensor.md): Represents a tensor of dates.

[`class HolidayCalendar`](../../tf_quant_finance/experimental/dates/HolidayCalendar.md): Represents a holiday calendar.

[`class HolidayCalendar2`](../../tf_quant_finance/experimental/dates/HolidayCalendar2.md): Represents a holiday calendar.

[`class Month`](../../tf_quant_finance/experimental/dates/Month.md): Months. Values are one-based.

[`class PeriodType`](../../tf_quant_finance/experimental/dates/PeriodType.md): Periods that can be added or subtracted from DateTensors.

[`class PeriodicSchedule`](../../tf_quant_finance/experimental/dates/PeriodicSchedule.md): Defines an array of dates specified by a regular schedule.

[`class WeekDay`](../../tf_quant_finance/experimental/dates/WeekDay.md): Named days of the week. Values are zero-based with Monday = 0.

[`class WeekendMask`](../../tf_quant_finance/experimental/dates/WeekendMask.md): Provides weekend masks for some of the common weekend patterns.

## Functions

[`convert_to_date_tensor(...)`](../../tf_quant_finance/experimental/dates/convert_to_date_tensor.md): Converts supplied data to a `DateTensor` if possible.

[`from_datetimes(...)`](../../tf_quant_finance/experimental/dates/from_datetimes.md): Creates DateTensor from a sequence of Python datetime objects.

[`from_np_datetimes(...)`](../../tf_quant_finance/experimental/dates/from_np_datetimes.md): Creates DateTensor from a Numpy array of dtype datetime64.

[`from_ordinals(...)`](../../tf_quant_finance/experimental/dates/from_ordinals.md): Creates DateTensor from tensors of ordinals.

[`from_tuples(...)`](../../tf_quant_finance/experimental/dates/from_tuples.md): Creates DateTensor from a sequence of year-month-day Tuples.

[`from_year_month_day(...)`](../../tf_quant_finance/experimental/dates/from_year_month_day.md): Creates DateTensor from tensors of years, months and days.

[`random_dates(...)`](../../tf_quant_finance/experimental/dates/random_dates.md): Generates random dates between the supplied start and end dates.

