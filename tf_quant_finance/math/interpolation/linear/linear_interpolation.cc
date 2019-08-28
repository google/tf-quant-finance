// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstddef>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace qf {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename Device, typename T>
struct LinearInterpolationFunctor {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstMatrix x,
                  typename TTypes<T>::ConstMatrix x_data,
                  typename TTypes<T>::ConstMatrix y_data,
                  typename TTypes<T>::ConstFlat left_slope,
                  typename TTypes<T>::ConstFlat right_slope,
                  typename TTypes<T>::Matrix output);
};

template <typename T>
struct LinearInterpolationFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, typename TTypes<T>::ConstMatrix x,
                  typename TTypes<T>::ConstMatrix x_data,
                  typename TTypes<T>::ConstMatrix y_data,
                  typename TTypes<T>::ConstFlat left_slope,
                  typename TTypes<T>::ConstFlat right_slope,
                  typename TTypes<T>::Matrix output) {
    const size_t batch_size = x.dimension(0);
    const size_t x_size = x.dimension(1);
    const size_t x_data_size = x_data.dimension(1);

    // This lambda expression does not live beyond the scope of this function as
    // it is only passed to blocking calls.
    const auto compute = [&](int start, int end) {
      size_t index = start % x_size;
      size_t batch = start / x_size;

      for (size_t i = start; i < end; ++i) {
        const T* x_data_begin_ptr = x_data.data() + batch * x_data_size;
        const T* x_data_end_ptr = x_data.data() + (batch + 1) * x_data_size;

        const T curr_x = x(batch, index);

        // Represent lower and upper bound of x(index) in the sorted x_data.
        // These bounds can be used to distinguish between edge cases (e.g.
        // extrapolation).
        auto bounds =
            std::equal_range(x_data_begin_ptr, x_data_end_ptr, curr_x);

        if (bounds.first == x_data_begin_ptr) {
          // Left extrapolation case (including the edge case where the x[index]
          // is equal to the leftmost point in x_data).
          output(batch, index) =
              left_slope(batch) * (curr_x - x_data(batch, 0)) +
              y_data(batch, 0);
        } else if (bounds.second == x_data_end_ptr) {
          // Right extrapolation case.
          output(batch, index) =
              right_slope(batch) * (curr_x - x_data(batch, x_data_size - 1)) +
              y_data(batch, x_data_size - 1);
        } else {
          const size_t left_index = bounds.first - x_data_begin_ptr;
          const size_t right_index = bounds.second - x_data_begin_ptr;

          if (left_index == right_index) {
            // Interpolation step.
            const T x1 = x_data(batch, left_index - 1);
            const T x2 = x_data(batch, left_index);
            const T y1 = y_data(batch, left_index - 1);
            const T y2 = y_data(batch, left_index);
            output(batch, index) = (y2 - y1) / (x2 - x1) * (curr_x - x1) + y1;
          } else {
            // Element x(index) is present in the x_data.
            output(batch, index) = y_data(batch, left_index);
          }
        }

        index++;
        if (index == x_size) {
          index = 0;
          ++batch;
        }
      }
    };

    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    const size_t num_elements = batch_size * x_size;
    const size_t num_threads = worker_threads.num_threads;

    // Each thread should process at least min_block_size elements.
    const size_t min_block_size = 32;
    const size_t block_size =
        std::max(num_elements / num_threads, min_block_size);

    worker_threads.workers->TransformRangeConcurrently(block_size, num_elements,
                                                       compute);
  }
};

}  // namespace functor

template <typename Device, typename T>
class LinearInterpolationOp : public OpKernel {
 public:
  explicit LinearInterpolationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& x_tensor = context->input(0);
    const Tensor& x_data_tensor = context->input(1);
    const Tensor& y_data_tensor = context->input(2);

    OP_REQUIRES(context,
                x_tensor.dims() == 2 && x_data_tensor.dims() == 2 &&
                    y_data_tensor.dims() == 2,
                Status(error::INVALID_ARGUMENT,
                       "x, x_data and y_data must be of rank 2"));

    // The first dimension is batch size.
    auto x = x_tensor.matrix<T>();
    auto x_data = x_data_tensor.matrix<T>();
    auto y_data = y_data_tensor.matrix<T>();

    OP_REQUIRES(context, x_data.dimensions() == y_data.dimensions(),
                Status(error::INVALID_ARGUMENT,
                       "x_data and y_data must be of the same shape"));

    const size_t batch_size = x.dimension(0);

    OP_REQUIRES(context, x_data.dimension(0) == batch_size,
                Status(error::INVALID_ARGUMENT,
                       "x, x_data and y_data must have the same batch size"));

    const auto l_sizes = context->input(3).shape().dim_sizes();
    const auto r_sizes = context->input(4).shape().dim_sizes();

    OP_REQUIRES(context, l_sizes.size() == 1 && l_sizes[0] == batch_size,
                errors::InvalidArgument(
                    "The shape of left_slope needs to match batching size"));

    OP_REQUIRES(context, r_sizes.size() == 1 && r_sizes[0] == batch_size,
                errors::InvalidArgument(
                    "The shape of right_slope needs to match batching size"));

    const auto left_slope = context->input(3).flat<T>();
    const auto right_slope = context->input(4).flat<T>();

    // Create and allocate memory for the output tensor.
    Tensor* output_ptr = nullptr;

    OP_REQUIRES_OK(context, context->allocate_output(
                                0, context->input(0).shape(), &output_ptr));

    auto output = output_ptr->matrix<T>();

    functor::LinearInterpolationFunctor<Device, T>()(
        context, x, x_data, y_data, left_slope, right_slope, output);
  }
};

REGISTER_OP("LinearInterpolation")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("x_data: T")
    .Input("y_data: T")
    .Input("left_slope: T")
    .Input("right_slope: T")
    .Output("y: T")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

REGISTER_KERNEL_BUILDER(
    Name("LinearInterpolation").Device(DEVICE_CPU).TypeConstraint<double>("T"),
    LinearInterpolationOp<CPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("LinearInterpolation").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    LinearInterpolationOp<CPUDevice, float>);

#if GOOGLE_CUDA

extern template struct functor::LinearInterpolationFunctor<GPUDevice, double>;
extern template struct functor::LinearInterpolationFunctor<GPUDevice, float>;

REGISTER_KERNEL_BUILDER(
    Name("LinearInterpolation").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    LinearInterpolationOp<GPUDevice, double>);
REGISTER_KERNEL_BUILDER(
    Name("LinearInterpolation").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    LinearInterpolationOp<GPUDevice, float>);

#endif  // GOOGLE_CUDA

}  // namespace qf
}  // namespace tensorflow
