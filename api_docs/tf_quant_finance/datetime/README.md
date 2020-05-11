# Tensorflow date and time module 

This module contains a Tensorflow-based implementation of date and time data
structures and utilities.

Currently only dates are supported, and the time-related parts will be added in
the future.

The module is structured in three layers:

- `DateTensor` and `PeriodTensor`.

These represent date and period tensors behaving much like regular Tensorflow
tensors. The are used for manipulating the Gregorian calendar. They don't
contain any finance-related knowledge, so can be used in any Tensorflow programs
that deal with dates. Under the hood, they use Tensorflow tensors, so that date
manipulations can be easily included as part of Tensorflow graphs.

Create `DateTensor`s using `dates_from_***` methods, and `PeriodTensor` using
methods such as `months()`. Many of the Tensorflow's array-manipulation ops are
implemented for these tensors, for example `my_date_tensor[2:5]`, 
`my_period_tensor.expand_dims(axis=2)`, with the ops involving multiple tensors
being class methods of corresponding classes, e.g.
`DateTensor.concat([date_tensor_1, date_tensor_2], axis=-1)`.

- `HolidayCalendar`.

This layer introduces the concept of a business day. Build a `HolidayCalendar`
using the `create_holiday_calendar` method by specifying weekends and
holidays (the library doesn't provide lists of holidays for specific countries,
these can be constructed e.g. using Pandas). The `HolidayCalendar` object can
then perform holiday-aware manipulations, such as advancing `DateTensor` by a
given number of business days or rolling to nearest business days according to
a given convention.

- Utilities.

These are built on top of the previous two layers and include such pieces as
schedule generation, random dates and day count conventions.
 