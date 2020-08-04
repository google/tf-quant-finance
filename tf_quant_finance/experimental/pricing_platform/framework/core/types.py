# Lint as: python3
# Copyright 2020 Google LLC
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
"""Common data types."""

import datetime
from typing import List, Union

import tensorflow.compat.v2 as tf
from tf_quant_finance import datetime as datetime_tff

from tf_quant_finance.experimental.pricing_platform.framework.core import business_days
from tf_quant_finance.experimental.pricing_platform.framework.core import currencies
from tf_quant_finance.experimental.pricing_platform.framework.core import daycount_conventions
from tf_quant_finance.experimental.pricing_platform.instrument_protos import period_pb2


# Typing can't resolve type of generated enums so we use Method Resolution Order
# to infer the type.
DayCountConventionsProtoType = daycount_conventions.DayCountConventionsProtoType
CurrencyProtoType = currencies.CurrencyProtoType
BankHolidaysProtoType = business_days.BankHolidaysProtoType
BusinessDayConventionProtoType = business_days.BusinessDayConventionProtoType
Period = Union[period_pb2.Period, datetime_tff.PeriodType]

Dtype = tf.compat.v1.DType
BoolTensor = tf.Tensor
IntTensor = tf.Tensor
FloatTensor = tf.Tensor
StringTensor = tf.Tensor
DateTensor = Union[datetime_tff.DateTensor, datetime.date, List[List[int]]]

GraphDef = tf.compat.v1.GraphDef

# Protobuf base type
ProtobufBaseType = period_pb2.__class__.__base__


__all__ = ["DayCountConventionsProtoType",
           "CurrencyProtoType",
           "BankHolidaysProtoType",
           "BusinessDayConventionProtoType",
           "Period",
           "Dtype",
           "BoolTensor",
           "IntTensor",
           "FloatTensor",
           "StringTensor",
           "DateTensor",
           "GraphDef",
           "ProtobufBaseType"]

