// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "embed_layer_norm.h"
#include "core/util/math_cpuonly.h"
#include "core/platform/threadpool.h"

#include "longformer_attention_base.h"

#include <atomic>

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      EmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      EmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)

Status LongformerAttentionBase__CheckInputs(const LongformerAttentionBase* p,
                                            const TensorShape& input_shape,
                                            const TensorShape& weights_shape,
                                            const TensorShape& bias_shape,
                                            const TensorShape& mask_shape,
                                            const TensorShape& global_weights_shape,
                                            const TensorShape& global_bias_shape,
                                            const TensorShape& global_shape) {
  return p->CheckInputs(input_shape,
                        weights_shape,
                        bias_shape,
                        mask_shape,
                        global_weights_shape,
                        global_bias_shape,
                        global_shape);
}

template <typename T>
EmbedLayerNorm<T>::EmbedLayerNorm(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &epsilon_).IsOK());
  ORT_ENFORCE(epsilon_ >= 0);
}

template <typename T>
EmbedLayerNorm<T>::Inputs::Inputs(const Tensor* input_ids,
                                  const Tensor* segment_ids,
                                  const Tensor* word_embedding,
                                  const Tensor* position_embedding,
                                  const Tensor* segment_embedding,
                                  const Tensor* gamma,
                                  const Tensor* beta,
                                  const Tensor* mask)
    : input_ids(input_ids)
    , segment_ids(segment_ids)
    , word_embedding(word_embedding)
    , position_embedding(position_embedding)
    , segment_embedding(segment_embedding)
    , gamma(gamma)
    , beta(beta)
    , mask(mask) {
  const auto& input_dims = input_ids->Shape().GetDims();

  batch_size = static_cast<int>(input_dims[0]);
  sequence_size = static_cast<int>(input_dims[1]);
  hidden_size = static_cast<int>(word_embedding->Shape()[1]);

  has_segment_embedding = segment_embedding != nullptr;

  word_embedding_length = static_cast<int>(word_embedding->Shape()[0]);
  position_embedding_length = static_cast<int>(position_embedding->Shape()[0]);
  segment_embedding_length = has_segment_embedding ? static_cast<int>(segment_embedding->Shape()[0]) : 0;
}

template <typename T>
Status EmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  Inputs inputs(/*input_ids=*/context->Input<Tensor>(0),
                /*segment_ids=*/context->Input<Tensor>(1),
                /*word_embedding=*/context->Input<Tensor>(2),
                /*position_embedding=*/context->Input<Tensor>(3),
                /*segment_embedding=*/context->Input<Tensor>(4),
                /*gamma=*/context->Input<Tensor>(5),
                /*beta=*/context->Input<Tensor>(6),
                /*mask=*/context->Input<Tensor>(7));

  ORT_RETURN_IF_ERROR(CheckInputs(inputs));
  return ComputeInternal(context, inputs);
}

//
// TODO(kreeger): left off right here - need to pass in some methods to modify the quantized stuff.
//
template <typename T>
Status EmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context,
                                          const Inputs& inputs) const {
  const int32_t* input_ids_data = inputs.input_ids->template Data<int32_t>();
  const int32_t* segment_ids_data =
    inputs.has_segment_embedding ? inputs.segment_ids->template Data<int32_t>() : nullptr;

  // Request outputs:
  TensorShape output_shape({inputs.batch_size, inputs.sequence_size, inputs.hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({inputs.batch_size});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  // TODO - helper method with type?
  // TODO - need to specify non-float32 here.
  const T* word_embedding_data = inputs.word_embedding->template Data<T>();
  const T* position_embedding_data = inputs.position_embedding->template Data<T>();
  const T* segment_embedding_data =
    inputs.has_segment_embedding ? inputs.segment_embedding->template Data<T>() : nullptr;
  const T* gamma_data = inputs.gamma->template Data<T>();
  const T* beta_data = inputs.beta->template Data<T>();
  T* output_data = output->template MutableData<T>();

  // One option - add virtual methods and call with a flag set on this base class.
  // TODO - figure out this template type stuff - might work with the fixes I made in Linux...

  // Calculate output
  {
    std::atomic_bool failed{false};

  int hidden_size = inputs.hidden_size;
    int n = inputs.batch_size * inputs.sequence_size;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
      int word_col_index = input_ids_data[index];
      if (word_col_index < 0 || word_col_index >= inputs.word_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int position_col_index = index % inputs.sequence_size;
      if (position_col_index >= inputs.position_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int segment_col_index = 0;
      if (nullptr != segment_ids_data) {
        segment_col_index = segment_ids_data[index];
        if (segment_col_index < 0 || segment_col_index >= inputs.segment_embedding_length) {
          failed.store(true, std::memory_order_release);
          return;
        }
      }

      T* y = output_data + index * hidden_size;
      const T* input_word_embedding = word_embedding_data + (static_cast<int64_t>(word_col_index) * hidden_size);
      const T* input_position_embedding =
          position_embedding_data + (static_cast<int64_t>(position_col_index) * hidden_size);
      const T* input_segment_embedding =
          (inputs.has_segment_embedding) ? segment_embedding_data + (static_cast<int64_t>(segment_col_index) * hidden_size) : nullptr;

      T sum = static_cast<T>(0);
      for (int i = 0; i < hidden_size; i++) {
        T subtotal = input_word_embedding[i] + input_position_embedding[i];
        if (inputs.has_segment_embedding) {
          subtotal += input_segment_embedding[i];
        }
        y[i] = subtotal;
        sum += subtotal;
      }
      T mean = sum / hidden_size;
      sum = 0;

      for (int i = 0; i < hidden_size; i++) {
        T a = y[i] - mean;
        y[i] = a;
        sum += a * a;
      }

      T e = sqrt(sum / hidden_size + static_cast<T>(epsilon_));
      for (int i = 0; i < hidden_size; i++) {
        y[i] = y[i] / e * gamma_data[i] + beta_data[i];
      }
    }, 0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != inputs.mask) {
    int sequence_size = inputs.sequence_size;
    const int32_t* mask_data = inputs.mask->template Data<int32_t>();
    for (int b = 0; b < inputs.batch_size; b++) {
      mask_index->template MutableData<int32_t>()[b] =
          static_cast<int32_t>(std::count_if(mask_data + (static_cast<int64_t>(b) * sequence_size),
                                             mask_data + (static_cast<int64_t>(b) * sequence_size) + sequence_size,
                                             [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, inputs.batch_size * sizeof(int32_t));
  }

  return Status::OK();
}

template <typename T>
Status EmbedLayerNorm<T>::CheckInputs(const Inputs& inputs) const {
  if (nullptr != inputs.segment_ids && inputs.input_ids->Shape() != inputs.segment_ids->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 1 shall have same shape");
  }

  if (nullptr != inputs.mask && inputs.input_ids->Shape() != inputs.mask->Shape()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 and 7 (mask) shall have same shape");
  }

  const auto& input_dims = inputs.input_ids->Shape().GetDims();
  if (input_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "input_ids is expected to have 2 dimensions, got ", input_dims.size());
  }

  const auto& word_embedding_dims = inputs.word_embedding->Shape().GetDims();
  if (word_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding is expected to have 2 dimensions, got ", word_embedding_dims.size());
  }

  const auto& position_embedding_dims = inputs.position_embedding->Shape().GetDims();
  if (position_embedding_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "position_embedding is expected to have 2 dimensions, got ", position_embedding_dims.size());
  }

  if (nullptr != inputs.segment_embedding) {
    const auto& segment_embedding_dims = inputs.segment_embedding->Shape().GetDims();
    if (segment_embedding_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "segment_embedding is expected to have 2 dimensions, got ", segment_embedding_dims.size());
    }
    if (word_embedding_dims[1] != segment_embedding_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "word_embedding and segment_embedding shall have same dimension 1");
    }
  }

  if (word_embedding_dims[1] != position_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "word_embedding and position_embedding shall have same dimension 1");
  }

  const auto& beta_dims = inputs.beta->Shape().GetDims();
  if (beta_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have 1 dimensions, got ", beta_dims.size());
  }

  if (beta_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "beta is expected to have size of ", word_embedding_dims[1], ", got ", beta_dims[0]);
  }

  const auto& gamma_dims = inputs.gamma->Shape().GetDims();
  if (gamma_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have 1 dimensions, got ", gamma_dims.size());
  }

  if (gamma_dims[0] != word_embedding_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "gamma is expected to have size of ", word_embedding_dims[1], ", got ", gamma_dims[0]);
  }

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
