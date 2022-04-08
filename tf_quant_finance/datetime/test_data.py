# Copyright 2019 Google LLC
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
"""Test data for tests in this package."""

import tf_quant_finance as tff

dates = tff.datetime

test_dates = [
    (1901, 1, 1),
    (1901, 2, 15),
    (1901, 2, 28),
    (1901, 3, 28),
    (2019, 1, 1),
    (2019, 1, 15),
    (2019, 1, 31),
    (2019, 2, 1),
    (2019, 2, 15),
    (2019, 2, 28),
    (2019, 3, 1),
    (2019, 3, 15),
    (2019, 3, 31),
    (2019, 4, 1),
    (2019, 4, 15),
    (2019, 4, 30),
    (2019, 5, 1),
    (2019, 5, 15),
    (2019, 5, 31),
    (2019, 6, 1),
    (2019, 6, 15),
    (2019, 6, 30),
    (2019, 7, 1),
    (2019, 7, 15),
    (2019, 7, 31),
    (2019, 8, 1),
    (2019, 8, 15),
    (2019, 8, 31),
    (2019, 9, 1),
    (2019, 9, 15),
    (2019, 9, 30),
    (2019, 10, 1),
    (2019, 10, 15),
    (2019, 10, 31),
    (2019, 11, 1),
    (2019, 11, 15),
    (2019, 11, 30),
    (2019, 12, 1),
    (2019, 12, 15),
    (2019, 12, 31),
    (2020, 1, 1),
    (2020, 1, 15),
    (2020, 1, 31),
    (2020, 2, 1),
    (2020, 2, 15),
    (2020, 2, 28),
    (2020, 2, 29),
    (2020, 3, 1),
    (2020, 3, 15),
    (2020, 3, 31),
    (2020, 4, 1),
    (2020, 4, 15),
    (2020, 4, 30),
    (2020, 5, 1),
    (2020, 5, 15),
    (2020, 5, 31),
    (2020, 6, 1),
    (2020, 6, 15),
    (2020, 6, 30),
    (2020, 7, 1),
    (2020, 7, 15),
    (2020, 7, 31),
    (2020, 8, 1),
    (2020, 8, 15),
    (2020, 8, 31),
    (2020, 9, 1),
    (2020, 9, 15),
    (2020, 9, 30),
    (2020, 10, 1),
    (2020, 10, 15),
    (2020, 10, 31),
    (2020, 11, 1),
    (2020, 11, 15),
    (2020, 11, 30),
    (2020, 12, 1),
    (2020, 12, 15),
    (2020, 12, 31),
    (2000, 2, 15),
    (2000, 2, 28),
    (2000, 2, 29),
    (2000, 3, 1),
    (2099, 3, 15),
    (2099, 2, 15),
    (2099, 2, 28),
    (2099, 12, 31),
    (1900, 2, 15),
    (1900, 2, 28),
    (1900, 3, 1),
    (2100, 2, 15),
    (2100, 2, 28),
    (2100, 3, 1),
    (2100, 3, 15),
    (2200, 2, 15),
    (2200, 2, 28),
    (2200, 3, 1),
    (2200, 3, 15),
    (2300, 2, 15),
    (2300, 2, 28),
    (2300, 3, 1),
    (2300, 3, 15),
]

day_addition_data = [
    ((2019, 2, 15), 5, (2019, 2, 20)),
    ((2019, 2, 15), 15, (2019, 3, 2)),
    ((2019, 2, 15), 365, (2020, 2, 15)),
    ((2019, 2, 15), 365 * 2, (2021, 2, 14)),
    ((2019, 2, 15), -5, (2019, 2, 10)),
    ((2019, 2, 15), -15, (2019, 1, 31)),
]

week_addition_data = [
    ((2019, 2, 15), 1, (2019, 2, 22)),
    ((2019, 2, 15), 3, (2019, 3, 8)),
    ((2019, 2, 15), 60, (2020, 4, 10)),
    ((2019, 2, 15), -2, (2019, 2, 1)),
]

month_addition_data = [
    ((2019, 2, 15), 1, (2019, 3, 15)),
    ((2019, 2, 15), 25, (2021, 3, 15)),
    ((2019, 1, 31), 3, (2019, 4, 30)),
    ((2019, 1, 31), 1, (2019, 2, 28)),
    ((2018, 11, 15), 1, (2018, 12, 15)),
    ((2018, 11, 15), 2, (2019, 1, 15)),
    ((2018, 11, 15), 13, (2019, 12, 15)),
    ((2018, 11, 15), 14, (2020, 1, 15)),
    ((2018, 11, 15), 16, (2020, 3, 15)),
    ((2018, 12, 15), 1, (2019, 1, 15)),
    ((2018, 12, 30), 2, (2019, 2, 28)),
    ((2018, 11, 29), 3, (2019, 2, 28)),
    ((2018, 11, 29), 15, (2020, 2, 29)),
    ((2020, 2, 29), 12, (2021, 2, 28)),
    ((2019, 2, 15), -1, (2019, 1, 15)),
    ((2019, 5, 31), -3, (2019, 2, 28)),
]

year_addition_data = [
    ((2019, 5, 15), 1, (2020, 5, 15)),
    ((2019, 7, 15), 25, (2044, 7, 15)),
    ((2020, 2, 29), 1, (2021, 2, 28)),
    ((2021, 2, 28), 1, (2022, 2, 28)),
    ((2020, 2, 29), 4, (2024, 2, 29)),
    ((2024, 2, 29), -5, (2019, 2, 28)),
]

day_of_year_data = [
    ((2019, 1, 1), 1),
    ((2019, 2, 15), 46),
    ((2019, 3, 10), 69),
    ((2019, 4, 8), 98),
    ((2019, 5, 20), 140),
    ((2019, 6, 24), 175),
    ((2019, 7, 9), 190),
    ((2019, 8, 12), 224),
    ((2019, 9, 30), 273),
    ((2019, 10, 11), 284),
    ((2019, 11, 28), 332),
    ((2019, 12, 31), 365),
    ((2020, 1, 1), 1),
    ((2020, 2, 15), 46),
    ((2020, 3, 10), 70),
    ((2020, 4, 8), 99),
    ((2020, 5, 20), 141),
    ((2020, 6, 24), 176),
    ((2020, 7, 9), 191),
    ((2020, 8, 12), 225),
    ((2020, 9, 30), 274),
    ((2020, 10, 11), 285),
    ((2020, 11, 28), 333),
    ((2020, 12, 31), 366),
]

invalid_dates = [
    (-5, 3, 15),
    (2015, -3, 15),
    (2015, 0, 15),
    (2015, 3, 0),
    (2015, 3, 32),
    (2015, 4, 31),
    (2015, 2, 29),
    (2016, 2, 30),
]

end_of_month_test_cases = [
    # (date, expected_is_end_of_month, expected_to_end_of_month)
    ((2019, 1, 30), False, (2019, 1, 31)),
    ((2019, 1, 31), True, (2019, 1, 31)),
    ((2019, 2, 15), False, (2019, 2, 28)),
    ((2019, 2, 28), True, (2019, 2, 28)),
    ((2020, 2, 28), False, (2020, 2, 29)),
    ((2020, 2, 29), True, (2020, 2, 29)),
    ((2018, 11, 5), False, (2018, 11, 30)),
    ((2018, 11, 30), True, (2018, 11, 30)),
]

holidays = [
    (2020, 1, 1),  # Wed
    (2020, 7, 3),  # Fri
    (2020, 12, 25),  # Fri
    (2021, 1, 1),  # Fri
    (2021, 7, 5),  # Mon
    (2021, 12, 24),  # Fri
]

is_business_day_data = [
    ((2020, 1, 1), False),
    ((2020, 1, 2), True),
    ((2020, 1, 3), True),
    ((2020, 1, 4), False),
    ((2020, 1, 5), False),
    ((2020, 3, 20), True),
    ((2020, 3, 21), False),
    ((2020, 3, 22), False),
    ((2020, 3, 23), True),
    ((2020, 7, 3), False),
    ((2020, 12, 25), False),
    ((2021, 1, 1), False),
    ((2021, 7, 5), False),
    ((2021, 12, 24), False),
    ((2021, 12, 25), False),
    ((2021, 12, 26), False),
    ((2021, 12, 31), True)
]

adjusted_dates_data = [
    {
        "date": (2020, 1, 4),  # Sat
        "unadjusted": (2020, 1, 4),
        "following": (2020, 1, 6),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 6),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2020, 1, 5),  # Sun
        "unadjusted": (2020, 1, 5),
        "following": (2020, 1, 6),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 6),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2020, 1, 3),  # Fri, business day
        "unadjusted": (2020, 1, 3),
        "following": (2020, 1, 3),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 3),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2020, 7, 3),  # Fri, holiday
        "unadjusted": (2020, 7, 3),
        "following": (2020, 7, 6),
        "preceding": (2020, 7, 2),
        "modified_following": (2020, 7, 6),
        "modified_preceding": (2020, 7, 2),
    },
    {
        "date": (2020, 7, 4),  # Sat, after holiday
        "unadjusted": (2020, 7, 4),
        "following": (2020, 7, 6),
        "preceding": (2020, 7, 2),
        "modified_following": (2020, 7, 6),
        "modified_preceding": (2020, 7, 2),
    },
    {
        "date": (2021, 2, 27),  # Sat
        "unadjusted": (2021, 2, 27),
        "following": (2021, 3, 1),
        "preceding": (2021, 2, 26),
        "modified_following": (2021, 2, 26),
        "modified_preceding": (2021, 2, 26),
    },
    {
        "date": (2021, 2, 28),  # Sun
        "unadjusted": (2021, 2, 28),
        "following": (2021, 3, 1),
        "preceding": (2021, 2, 26),
        "modified_following": (2021, 2, 26),
        "modified_preceding": (2021, 2, 26),
    },
    {
        "date": (2020, 2, 1),  # Sat
        "unadjusted": (2020, 2, 1),
        "following": (2020, 2, 3),
        "preceding": (2020, 1, 31),
        "modified_following": (2020, 2, 3),
        "modified_preceding": (2020, 2, 3),
    },
]

add_months_data = [
    {
        "date": (2019, 12, 4),
        "months": 1,
        "unadjusted": (2020, 1, 4),
        "following": (2020, 1, 6),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 6),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2019, 11, 5),
        "months": 2,
        "unadjusted": (2020, 1, 5),
        "following": (2020, 1, 6),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 6),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2019, 10, 3),
        "months": 3,
        "unadjusted": (2020, 1, 3),
        "following": (2020, 1, 3),
        "preceding": (2020, 1, 3),
        "modified_following": (2020, 1, 3),
        "modified_preceding": (2020, 1, 3),
    },
    {
        "date": (2020, 2, 3),
        "months": 5,
        "unadjusted": (2020, 7, 3),
        "following": (2020, 7, 6),
        "preceding": (2020, 7, 2),
        "modified_following": (2020, 7, 6),
        "modified_preceding": (2020, 7, 2),
    },
    {
        "date": (2020, 2, 29),
        "months": 12,
        "unadjusted": (2021, 2, 28),
        "following": (2021, 3, 1),
        "preceding": (2021, 2, 26),
        "modified_following": (2021, 2, 26),
        "modified_preceding": (2021, 2, 26),
    },
    {
        "date": (2020, 2, 1),
        "months": 6,
        "unadjusted": (2020, 8, 1),
        "following": (2020, 8, 3),
        "preceding": (2020, 7, 31),
        "modified_following": (2020, 8, 3),
        "modified_preceding": (2020, 8, 3),
    },
]

add_days_data = [  # Assumes "modified following" convention.
    {
        "date": (2020, 1, 2),
        "days": 1,
        "shifted_date": (2020, 1, 3),
    },
    {
        "date": (2020, 1, 2),
        "days": 2,
        "shifted_date": (2020, 1, 6),
    },
    {
        "date": (2020, 1, 2),
        "days": 10,
        "shifted_date": (2020, 1, 16),
    },
    {
        "date": (2020, 1, 4),   # Sat
        "days": 0,
        "shifted_date": (2020, 1, 6),
    },
    {
        "date": (2020, 1, 4),
        "days": 1,
        "shifted_date": (2020, 1, 7),
    },
    {
        "date": (2020, 1, 4),
        "days": -1,
        "shifted_date": (2020, 1, 3),
    },
    {
        "date": (2020, 2, 27),
        "days": 3,
        "shifted_date": (2020, 3, 3),
    },
    {
        "date": (2021, 2, 26),
        "days": 3,
        "shifted_date": (2021, 3, 3),
    },
    {
        "date": (2020, 2, 29),   # Sat
        "days": 0,
        "shifted_date": (2020, 2, 28),
    },
    {
        "date": (2020, 2, 29),
        "days": 1,
        "shifted_date": (2020, 3, 2),
    },
    {
        "date": (2020, 2, 29),
        "days": -1,
        "shifted_date": (2020, 2, 27),
    },
    {
        "date": (2020, 6, 29),
        "days": 5,
        "shifted_date": (2020, 7, 7),
    },
    {
        "date": (2020, 6, 29),
        "days": 10,
        "shifted_date": (2020, 7, 14),
    },
    {
        "date": (2020, 12, 23),
        "days": 10,
        "shifted_date": (2021, 1, 8),
    },
    {
        "date": (2021, 12, 29),
        "days": 2,
        "shifted_date": (2021, 12, 31),
    }
]

days_between_data = [
    {
        "date1": (2020, 1, 2),
        "date2": (2020, 1, 3),
        "days": 1,
    },
    {
        "date1": (2020, 1, 2),
        "date2": (2020, 1, 6),
        "days": 2,
    },
    {
        "date1": (2020, 1, 6),
        "date2": (2020, 1, 2),
        "days": 0,
    },
    {
        "date1": (2020, 1, 6),
        "date2": (2020, 1, 6),
        "days": 0,
    },
    {
        "date1": (2020, 1, 2),
        "date2": (2020, 1, 16),
        "days": 10,
    },
    {
        "date1": (2020, 1, 3),
        "date2": (2020, 1, 4),
        "days": 1,
    },
    {
        "date1": (2020, 1, 4),
        "date2": (2020, 1, 6),
        "days": 0,
    },
    {
        "date1": (2020, 1, 4),
        "date2": (2020, 1, 5),
        "days": 0,
    },
    {
        "date1": (2020, 1, 4),
        "date2": (2020, 1, 7),
        "days": 1,
    },
    {
        "date1": (2020, 1, 4),
        "date2": (2020, 1, 3),
        "days": 0,
    },
    {
        "date1": (2020, 2, 27),
        "days": 3,
        "date2": (2020, 3, 3),
    },
    {
        "date1": (2021, 2, 26),
        "days": 3,
        "date2": (2021, 3, 3),
    },
    {
        "date1": (2020, 2, 29),   # Sat
        "date2": (2020, 2, 28),
        "days": 0,
    },
    {
        "date1": (2020, 2, 29),
        "date2": (2020, 3, 1),
        "days": 0,
    },
    {
        "date1": (2020, 2, 29),
        "date2": (2020, 3, 2),
        "days": 0,
    },
    {
        "date1": (2020, 2, 29),
        "date2": (2020, 3, 3),
        "days": 1,
    },
    {
        "date1": (2020, 6, 29),
        "date2": (2020, 7, 7),
        "days": 5,
    },
    {
        "date1": (2020, 6, 29),
        "date2": (2020, 7, 14),
        "days": 10,
    },
    {
        "date1": (2020, 12, 23),
        "date2": (2021, 1, 8),
        "days": 10,
    },
    {
        "date1": (2021, 12, 29),
        "date2": (2021, 12, 31),
        "days": 2,
    }
]

# Assumes calendar with only weekends and modified-following convention.
# end_of_month is False by default.
periodic_schedule_dynamic = [
    {
        "testcase_name": "monthly_forward",
        "start_dates": [737425],
        "end_dates": [737880],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2020, 1, 1), (2020, 2, 3), (2020, 3, 2),
                               (2020, 4, 1), (2020, 5, 1), (2020, 6, 1),
                               (2020, 7, 1), (2020, 8, 3), (2020, 9, 1),
                               (2020, 10, 1), (2020, 11, 2), (2020, 12, 1),
                               (2021, 1, 1), (2021, 2, 1), (2021, 3, 1),
                               (2021, 3, 31)]]
    },
    {
        "testcase_name": "yearly",
        "start_dates": [738945],
        "end_dates": [740712],
        "period_type": dates.PeriodType.YEAR,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2024, 2, 29), (2025, 2, 28), (2026, 2, 27),
                               (2027, 2, 26), (2028, 2, 29), (2028, 12, 29)]]
    }]

periodic_schedule_test_cases = [
    {
        "testcase_name": "monthly_forward",
        "start_dates": [(2020, 1, 1)],
        "end_dates": [(2021, 3, 31)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2020, 1, 1), (2020, 2, 3), (2020, 3, 2),
                               (2020, 4, 1), (2020, 5, 1), (2020, 6, 1),
                               (2020, 7, 1), (2020, 8, 3), (2020, 9, 1),
                               (2020, 10, 1), (2020, 11, 2), (2020, 12, 1),
                               (2021, 1, 1), (2021, 2, 1), (2021, 3, 1),
                               (2021, 3, 31)]]
    },
    {
        "testcase_name": "monthly_backward",
        "start_dates": [(2020, 1, 1)],
        "end_dates": [(2021, 3, 31)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 1,
        "backward": True,
        "expected_schedule": [[(2020, 1, 1), (2020, 1, 31), (2020, 2, 28),
                               (2020, 3, 31), (2020, 4, 30), (2020, 5, 29),
                               (2020, 6, 30), (2020, 7, 31), (2020, 8, 31),
                               (2020, 9, 30), (2020, 10, 30), (2020, 11, 30),
                               (2020, 12, 31), (2021, 1, 29), (2021, 2, 26),
                               (2021, 3, 31)]]
    },
    {
        "testcase_name": "quarterly_forward",
        "start_dates": [(2020, 1, 15)],
        "end_dates": [(2021, 3, 31)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 3,
        "backward": False,
        "expected_schedule": [[(2020, 1, 15), (2020, 4, 15), (2020, 7, 15),
                               (2020, 10, 15), (2021, 1, 15), (2021, 3, 31)]]
    },
    {
        "testcase_name": "quarterly_backward",
        "start_dates": [(2020, 1, 15)],
        "end_dates": [(2021, 3, 25)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 3,
        "backward": True,
        "expected_schedule": [[(2020, 1, 15), (2020, 3, 25), (2020, 6, 25),
                               (2020, 9, 25), (2020, 12, 25), (2021, 3, 25)]]
    },
    {
        "testcase_name": "yearly",
        "start_dates": [(2024, 2, 29)],
        "end_dates": [(2028, 12, 31)],
        "period_type": dates.PeriodType.YEAR,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2024, 2, 29), (2025, 2, 28), (2026, 2, 27),
                               (2027, 2, 26), (2028, 2, 29), (2028, 12, 29)]]
    },
    {
        "testcase_name": "biweekly",
        "start_dates": [(2020, 11, 20)],
        "end_dates": [(2021, 1, 31)],
        "period_type": dates.PeriodType.WEEK,
        "period_quantities": 2,
        "backward": False,
        "expected_schedule": [[(2020, 11, 20), (2020, 12, 4), (2020, 12, 18),
                               (2021, 1, 1), (2021, 1, 15), (2021, 1, 29),
                               (2021, 1, 29)]]
    },
    {
        "testcase_name": "every_10_days",
        "start_dates": [(2020, 5, 1)],
        "end_dates": [(2020, 7, 1)],
        "period_type": dates.PeriodType.DAY,
        "period_quantities": 10,
        "backward": False,
        "expected_schedule": [[(2020, 5, 1), (2020, 5, 11), (2020, 5, 21),
                               (2020, 5, 29), (2020, 6, 10), (2020, 6, 22),
                               (2020, 6, 30), (2020, 7, 1)]]
    },
    {
        "testcase_name": "includes_end_date_if_on_schedule",
        "start_dates": [(2020, 11, 20)],
        "end_dates": [(2021, 1, 29)],
        "period_type": dates.PeriodType.WEEK,
        "period_quantities": 2,
        "backward": False,
        "expected_schedule": [[(2020, 11, 20), (2020, 12, 4), (2020, 12, 18),
                               (2021, 1, 1), (2021, 1, 15), (2021, 1, 29)]]
    },
    {
        "testcase_name": "includes_start_date_if_on_schedule",
        "start_dates": [(2020, 3, 25)],
        "end_dates": [(2021, 3, 25)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 3,
        "backward": True,
        "expected_schedule": [[(2020, 3, 25), (2020, 6, 25), (2020, 9, 25),
                               (2020, 12, 25), (2021, 3, 25)]]
    },
    {
        "testcase_name": "rolls_start_date_out_of_bounds",
        "start_dates": [(2020, 2, 29)],
        "end_dates": [(2024, 12, 31)],
        "period_type": dates.PeriodType.YEAR,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2020, 2, 28), (2021, 2, 26), (2022, 2, 28),
                               (2023, 2, 28), (2024, 2, 29), (2024, 12, 31)]]
    },
    {
        "testcase_name": "rolls_end_date_out_of_bounds",
        "start_dates": [(2020, 1, 1)],
        "end_dates": [(2020, 2, 2)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 1,
        "backward": False,
        "expected_schedule": [[(2020, 1, 1), (2020, 2, 3), (2020, 2, 3)]]
    },
    {
        "testcase_name": "batch_with_same_period",
        "start_dates": [(2020, 1, 15), (2020, 4, 15)],
        "end_dates": [(2021, 3, 31), (2021, 1, 1)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": 3,
        "backward": False,
        "expected_schedule": [[(2020, 1, 15), (2020, 4, 15), (2020, 7, 15),
                               (2020, 10, 15), (2021, 1, 15), (2021, 3, 31)],
                              [(2020, 4, 15), (2020, 7, 15), (2020, 10, 15),
                               (2021, 1, 1), (2021, 1, 1), (2021, 1, 1)]]
    },
    {
        "testcase_name": "batch_with_different_periods",
        "start_dates": [(2020, 1, 15), (2020, 4, 15)],
        "end_dates": [(2021, 3, 31), (2021, 1, 1)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": [4, 6],
        "backward": False,
        "expected_schedule": [[(2020, 1, 15), (2020, 5, 15), (2020, 9, 15),
                               (2021, 1, 15), (2021, 3, 31)],
                              [(2020, 4, 15), (2020, 10, 15), (2021, 1, 1),
                               (2021, 1, 1), (2021, 1, 1)]]
    },
    {
        "testcase_name": "batch_backward",
        "start_dates": [(2020, 1, 15), (2020, 4, 15)],
        "end_dates": [(2021, 3, 31), (2021, 1, 1)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": [4, 6],
        "backward": True,
        "expected_schedule": [[(2020, 1, 15), (2020, 3, 31), (2020, 7, 31),
                               (2020, 11, 30), (2021, 3, 31)],
                              [(2020, 4, 15), (2020, 4, 15), (2020, 4, 15),
                               (2020, 7, 1), (2021, 1, 1)]]
    },
    {
        "testcase_name": "rank_2_batch",
        "start_dates": [[(2020, 1, 15), (2020, 4, 15)],
                        [(2020, 1, 17), (2020, 5, 12)]],
        "end_dates": [[(2020, 12, 31), (2020, 11, 30)],
                      [(2020, 10, 31), (2020, 9, 30)]],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": [[4, 3], [3, 2]],
        "backward": False,
        "expected_schedule": [[[(2020, 1, 15), (2020, 5, 15), (2020, 9, 15),
                                (2020, 12, 31), (2020, 12, 31)],
                               [(2020, 4, 15), (2020, 7, 15), (2020, 10, 15),
                                (2020, 11, 30), (2020, 11, 30)]],
                              [[(2020, 1, 17), (2020, 4, 17), (2020, 7, 17),
                                (2020, 10, 19), (2020, 10, 30)],
                               [(2020, 5, 12), (2020, 7, 13), (2020, 9, 14),
                                (2020, 9, 30), (2020, 9, 30)]]]
    },
    {
        "testcase_name": "end_of_month_forward",
        "start_dates": [(2020, 1, 15), (2020, 4, 30)],
        "end_dates": [(2021, 3, 31), (2021, 1, 1)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": [4, 3],
        "backward": False,
        "end_of_month": True,
        "expected_schedule": [[(2020, 1, 15), (2020, 5, 15), (2020, 9, 15),
                               (2021, 1, 15), (2021, 3, 31)],
                              [(2020, 4, 30), (2020, 7, 31), (2020, 10, 30),
                               (2021, 1, 29), (2021, 1, 29)]]
    },
    {
        "testcase_name": "end_of_month_backward",
        "start_dates": [(2020, 1, 15), (2020, 4, 15)],
        # Note that (2021, 2, 28) is Sunday, so it's rolled, but we still apply
        # end_of_month.
        "end_dates": [(2021, 2, 28), (2021, 1, 1)],
        "period_type": dates.PeriodType.MONTH,
        "period_quantities": [4, 6],
        "backward": True,
        "end_of_month": True,
        "expected_schedule": [[(2020, 1, 31), (2020, 2, 28), (2020, 6, 30),
                               (2020, 10, 30), (2021, 2, 26)],
                              [(2020, 4, 15), (2020, 4, 15), (2020, 4, 15),
                               (2020, 7, 1), (2021, 1, 1)]]
    },
]

days_in_leap_years_test_cases = [
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 2, 19),
        "expected": 2
    },
    {
        "date1": (2019, 2, 17),
        "date2": (2019, 2, 19),
        "expected": 0
    },
    {
        "date1": (2100, 2, 17),
        "date2": (2100, 2, 19),
        "expected": 0
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 4, 5),
        "expected": 13 + 31 + 4
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 12, 31),
        "expected": 318  # 366 - 31 - 17
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2021, 1, 1),
        "expected": 319
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2021, 5, 12),
        "expected": 319
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2023, 12, 31),
        "expected": 319
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2024, 1, 1),
        "expected": 319
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2024, 1, 5),
        "expected": 323
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2024, 12, 31),
        "expected": 684
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2025, 1, 1),
        "expected": 685
    },
    {
        "date1": (2020, 1, 1),
        "date2": (2025, 1, 1),
        "expected": 732
    },
    {
        "date1": (2019, 12, 31),
        "date2": (2025, 1, 1),
        "expected": 732
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2025, 1, 1),
        "expected": 732
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2024, 2, 10),
        "expected": 406  # 366 + 31 + 9
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2024, 1, 1),
        "expected": 366
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2022, 4, 11),
        "expected": 366
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2020, 12, 31),
        "expected": 365
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2020, 2, 10),
        "expected": 40
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2020, 1, 1),
        "expected": 0
    },
    {
        "date1": (2019, 8, 24),
        "date2": (2019, 10, 17),
        "expected": 0
    },
    {
        "date1": (2020, 12, 31),
        "date2": (2021, 10, 17),
        "expected": 1
    },
    {
        "date1": (2100, 12, 31),
        "date2": (2101, 10, 17),
        "expected": 0
    },
    {
        "date1": (2100, 5, 3),
        "date2": (2100, 12, 31),
        "expected": 0
    },
    {
        "date1": (2096, 12, 15),
        "date2": (2100, 12, 31),
        "expected": 17
    },
    {
        "date1": (2096, 12, 15),
        "date2": (2104, 1, 4),
        "expected": 20
    },
    {
        "date1": (2020, 2, 19),
        "date2": (2020, 2, 17),
        "expected": -2
    },
    {
        "date1": (2024, 2, 10),
        "date2": (2019, 8, 24),
        "expected": -406
    },
]

leap_days_between_dates_test_cases = [
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 2, 19),
        "expected": 0
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 3, 19),
        "expected": 1
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 3, 1),
        "expected": 1
    },
    {
        "date1": (2020, 2, 17),
        "date2": (2020, 2, 29),
        "expected": 0
    },
    {
        "date1": (2019, 2, 17),
        "date2": (2019, 3, 19),
        "expected": 0
    },
    {
        "date1": (2100, 2, 17),
        "date2": (2100, 3, 19),
        "expected": 0
    },
    {
        "date1": (2000, 2, 17),
        "date2": (2000, 3, 19),
        "expected": 1
    },
    {
        "date1": (2019, 2, 17),
        "date2": (2024, 3, 1),
        "expected": 2
    },
    {
        "date1": (2020, 2, 29),
        "date2": (2024, 3, 1),
        "expected": 2
    },
    {
        "date1": (2020, 3, 1),
        "date2": (2024, 3, 1),
        "expected": 1
    },
    {
        "date1": (2003, 3, 1),
        "date2": (2024, 3, 1),
        "expected": 6
    },
    {
        "date1": (2003, 3, 1),
        "date2": (2103, 3, 1),
        "expected": 24
    },
]

business_day_schedule_test_cases = [
    {
        "testcase_name": "within_business_week",
        "start_dates": [(2020, 3, 17)],
        "end_dates": [(2020, 3, 20)],
        "holidays": [],
        "backward": False,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 18), (2020, 3, 19),
                               (2020, 3, 20)]]
    },
    {
        "testcase_name": "across_weekend",
        "start_dates": [(2020, 3, 17)],
        "end_dates": [(2020, 3, 24)],
        "holidays": [],
        "backward": False,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 18), (2020, 3, 19),
                               (2020, 3, 20), (2020, 3, 23), (2020, 3, 24)]]
    },
    {
        "testcase_name": "across_holidays",
        "start_dates": [(2020, 3, 17)],
        "end_dates": [(2020, 3, 24)],
        "holidays": [(2020, 3, 18), (2020, 3, 20)],
        "backward": False,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 19),
                               (2020, 3, 23), (2020, 3, 24)]]
    },
    {
        "testcase_name": "ending_on_weekend",
        "start_dates": [(2020, 3, 17)],
        "end_dates": [(2020, 3, 22)],
        "holidays": [],
        "backward": False,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 18), (2020, 3, 19),
                               (2020, 3, 20)]]
    },
    {
        "testcase_name": "starting_on_weekend",
        "start_dates": [(2020, 3, 14)],
        "end_dates": [(2020, 3, 19)],
        "holidays": [],
        "backward": False,
        "expected_schedule": [[(2020, 3, 16), (2020, 3, 17), (2020, 3, 18),
                               (2020, 3, 19)]]
    },
    {
        "testcase_name": "batch_forward",
        "start_dates": [(2020, 3, 17), (2020, 3, 14)],
        "end_dates": [(2020, 3, 24), (2020, 3, 19)],
        "holidays": [],
        "backward": False,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 18), (2020, 3, 19),
                               (2020, 3, 20), (2020, 3, 23), (2020, 3, 24)],
                              [(2020, 3, 16), (2020, 3, 17), (2020, 3, 18),
                               (2020, 3, 19), (2020, 3, 19), (2020, 3, 19)]]
    },
    {
        "testcase_name": "batch_backward",
        "start_dates": [(2020, 3, 17), (2020, 3, 14)],
        "end_dates": [(2020, 3, 24), (2020, 3, 19)],
        "holidays": [],
        "backward": True,
        "expected_schedule": [[(2020, 3, 17), (2020, 3, 18), (2020, 3, 19),
                               (2020, 3, 20), (2020, 3, 23), (2020, 3, 24)],
                              [(2020, 3, 16), (2020, 3, 16), (2020, 3, 16),
                               (2020, 3, 17), (2020, 3, 18), (2020, 3, 19)]]
    },
]

