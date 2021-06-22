// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qembed_layer_norm.h"

#include <cmath>

#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

namespace {

// TODO(kreeger): Find a global home for these helper methods.
template <typename T>
inline T GetQuantizedInputTensorValue(const Tensor* tensor) {
  return *(tensor->template Data<T>());
}

template <typename T>
inline float Dequantize(T value, float scale, T zero_point) {
  return static_cast<float>(static_cast<int32_t>(value) - zero_point) * scale;
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QEmbedLayerNormalization,                                    \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QEmbedLayerNorm<T>);

REGISTER_KERNEL_TYPED(float)


template <typename T>
QEmbedLayerNorm<T>::QEmbedLayerNorm(const OpKernelInfo& op_kernel_info)
    : EmbedLayerNorm<T>(op_kernel_info) {
}

template <typename T>
QEmbedLayerNorm<T>::QInputs::QInputs(const Tensor* input_ids,
                                     const Tensor* segment_ids,
                                     const Tensor* word_embedding,
                                     const Tensor* position_embedding,
                                     const Tensor* segment_embedding,
                                     const Tensor* gamma,
                                     const Tensor* beta,
                                     const Tensor* word_embedding_scale,
                                     const Tensor* position_embedding_scale,
                                     const Tensor* segment_embedding_scale,
                                     const Tensor* gamma_scale,
                                     const Tensor* beta_scale,
                                     const Tensor* word_embedding_zero_point,
                                     const Tensor* position_embedding_zero_point,
                                     const Tensor* segment_embedding_zero_point,
                                     const Tensor* gamma_zero_point,
                                     const Tensor* beta_zero_point,
                                     const Tensor* mask)
    : EmbedLayerNorm<T>::Inputs(input_ids,
             segment_ids,
             word_embedding,
             position_embedding,
             segment_embedding,
             gamma,
             beta,
             mask)
    , word_embedding_scale(word_embedding_scale)
    , position_embedding_scale(position_embedding_scale)
    , segment_embedding_scale(segment_embedding_scale)
    , gamma_scale(gamma_scale)
    , beta_scale(beta_scale)
    , word_embedding_zero_point(word_embedding_zero_point)
    , position_embedding_zero_point(position_embedding_zero_point)
    , segment_embedding_zero_point(segment_embedding_zero_point)
    , gamma_zero_point(gamma_zero_point)
    , beta_zero_point(beta_zero_point) {}

template <typename T>
Status QEmbedLayerNorm<T>::Compute(OpKernelContext* context) const {
  QInputs inputs(/*input_ids=*/context->Input<Tensor>(0),
                 /*segment_ids=*/context->Input<Tensor>(1),
                 /*word_embedding=*/context->Input<Tensor>(2),
                 /*position_embedding=*/context->Input<Tensor>(3),
                 /*segment_embedding=*/context->Input<Tensor>(4),
                 /*gamma=*/context->Input<Tensor>(5),
                 /*beta=*/context->Input<Tensor>(6),
                 /*word_embedding_scale=*/context->Input<Tensor>(7),
                 /*position_embedding_scaled=*/context->Input<Tensor>(8),
                 /*segment_embedding_scale=*/context->Input<Tensor>(9),
                 /*gamma_scale=*/context->Input<Tensor>(10),
                 /*beta_scale=*/context->Input<Tensor>(11),
                 /*word_embedding_zero_point=*/context->Input<Tensor>(12),
                 /*position_embedding_zero_point=*/context->Input<Tensor>(13),
                 /*segment_embedding_zero_point=*/context->Input<Tensor>(14),
                 /*gamma_zero_point=*/context->Input<Tensor>(15),
                 /*beta_zero_point=*/context->Input<Tensor>(16),
                 /*mask=*/context->Input<Tensor>(17));

  ORT_RETURN_IF_ERROR(CheckInputs(inputs));
  return ComputeInternal(context, inputs);
}

template <typename T>
Status QEmbedLayerNorm<T>::CheckInputs(const QInputs& inputs) const {
  ORT_RETURN_IF_ERROR(EmbedLayerNorm<T>::CheckInputs(inputs));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.word_embedding_scale),
      "Word embedding scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.position_embedding_scale),
      "Position embedding scale must be a scalar or 1D tensor of size 1");
  if (inputs.segment_embedding != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.segment_embedding_scale),
        "Segment embedding scale must be a scalar or 1D tensor of size 1");
  }
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.gamma_scale),
      "Gamma scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.beta_scale),
      "Beta scale must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.word_embedding_zero_point),
      "Word embedding zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.position_embedding_zero_point),
      "Position embedding zero point must be a scalar or 1D tensor of size 1");
  if (inputs.segment_embedding != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.segment_embedding_zero_point),
                      "Segment embedding zero point must be a scalar or 1D tensor of size 1");
  }
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.gamma_zero_point),
      "Gamma zero point must be a scalar or 1D tensor of size 1");
  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(inputs.beta_zero_point),
      "Beta zero point must be a scalar or 1D tensor of size 1");

  return Status::OK();
}

template <typename T>
Status QEmbedLayerNorm<T>::ComputeInternal(OpKernelContext* context, const QInputs& inputs) const {
  // Determine shapes
  const auto& input_dims = inputs.input_ids->Shape().GetDims();
  int64_t hidden_size = inputs.word_embedding->Shape()[1];

  int batch_size = static_cast<int>(input_dims[0]);
  int sequence_length = static_cast<int>(input_dims[1]);

  // Segment inputs are optional and nullptr if this model is distill-bert:
  bool has_segment_embedding = inputs.segment_embedding != nullptr;

  int word_embedding_length = static_cast<int>(inputs.word_embedding->Shape()[0]);
  int position_embedding_length = static_cast<int>(inputs.position_embedding->Shape()[0]);
  int segment_embedding_length =
      has_segment_embedding ? static_cast<int>(inputs.segment_embedding->Shape()[0]) : 0;

  // Grab quantization values:
  float word_embedding_scale = *(inputs.word_embedding_scale->template Data<float>());
  uint8_t word_embedding_zero_point =
    *(inputs.word_embedding_zero_point->template Data<uint8_t>()); 

  float position_embedding_scale = *(inputs.position_embedding_scale->template Data<float>());
  uint8_t position_embedding_zero_point =
    *(inputs.position_embedding_zero_point->template Data<uint8_t>());

  float segment_embedding_scale =
    has_segment_embedding ? *(inputs.segment_embedding_scale->template Data<float>()) : 0.0f;
  uint8_t segment_embedding_zero_point =
      has_segment_embedding ? *(inputs.segment_embedding_zero_point->template Data<uint8_t>()) : 0;

  float layer_norm_weights_scale = *(inputs.gamma_scale->template Data<float>());
  uint8_t layer_norm_weights_zero_point = *(inputs.gamma_zero_point->template Data<uint8_t>());

  float layer_norm_bias_scale = *(inputs.beta_scale->template Data<float>());
  uint8_t layer_norm_bias_zero_point = *(inputs.beta_zero_point->template Data<uint8_t>());

  /*
  Output Tensors List:
  [0] layernorm_out (T)
  [1] mask_index_out (int32)
  */
  TensorShape output_shape({input_dims[0], input_dims[1], hidden_size});
  Tensor* output = context->Output(0, output_shape);

  TensorShape mask_index_shape({input_dims[0]});
  Tensor* mask_index = context->Output(1, mask_index_shape);

  // Grab pointers to buffers each Tensor represents:
  const int32_t* input_ids_data = inputs.input_ids->template Data<int32_t>();
  const int32_t* segment_ids_data =
      has_segment_embedding ? inputs.segment_ids->template Data<int32_t>() : nullptr;
  const uint8_t* word_embedding_data = inputs.word_embedding->template Data<uint8_t>();
  const uint8_t* position_embedding_data = inputs.position_embedding->template Data<uint8_t>();
  const uint8_t* segment_embedding_data =
      has_segment_embedding ? inputs.segment_embedding->template Data<uint8_t>() : nullptr;
  const uint8_t* gamma_data = inputs.gamma->template Data<uint8_t>();
  const uint8_t* beta_data = inputs.beta->template Data<uint8_t>();

  T* output_data = output->template MutableData<T>();

  // TODO(kreeger): consider using std::function<> here to reuse this code w/ the floating
  //                point version. See qlinear_binary_op_test.cc:~141
  // Perform the Op:
  {
    std::atomic_bool failed{false};

    int n = batch_size * sequence_length;
    concurrency::ThreadPool::TryBatchParallelFor(
        context->GetOperatorThreadPool(), n, [=, &failed](ptrdiff_t index) {
      int word_col_index = input_ids_data[index];
      if (word_col_index < 0 || word_col_index >= word_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int position_col_index = index % sequence_length;
      if (position_col_index >= position_embedding_length) {
        failed.store(true, std::memory_order_release);
        return;
      }
      int segment_col_index = 0;
      if (nullptr != segment_ids_data) {
        segment_col_index = segment_ids_data[index];
        if (segment_col_index < 0 || segment_col_index >= segment_embedding_length) {
          failed.store(true, std::memory_order_release);
          return;
        }
      }

      // Grab inputs for the embeddings for the current batch index:
      const uint8_t* input_word_embedding = word_embedding_data + (word_col_index * hidden_size);
      const uint8_t* input_position_embedding =
          position_embedding_data + (position_col_index * hidden_size);
      const uint8_t* input_segment_embedding = nullptr;
      if (segment_embedding_data != nullptr) {
        input_segment_embedding = segment_embedding_data + (segment_col_index * hidden_size);
      }

      T* output = output_data + (index * hidden_size);

      T sum = static_cast<T>(0);
      for (int i = 0; i < hidden_size; ++i) {
        // pass a lambda for these dequantize calls.
        T subtotal = Dequantize(input_word_embedding[i],
                                word_embedding_scale,
                                word_embedding_zero_point) +
                     Dequantize(input_position_embedding[i],
                                position_embedding_scale,
                                position_embedding_zero_point);
        if (segment_embedding_data != nullptr) {
          subtotal += Dequantize(input_segment_embedding[i],
                                 segment_embedding_scale,
                                 segment_embedding_zero_point);
        }
        output[i] = subtotal;
        sum += subtotal;
      }

      T mean = sum / hidden_size;
      sum = 0;

      for (int i = 0; i < hidden_size; i++) {
        T a = output[i] - mean;
        output[i] = a;
        sum += a * a;
      }

      T e = sqrt(sum / hidden_size + static_cast<T>(EmbedLayerNorm<T>::epsilon_));
      for (int i = 0; i < hidden_size; i++) {
        T cur_gamma = Dequantize(gamma_data[i],
                                  layer_norm_weights_scale,
                                  layer_norm_weights_zero_point);
        T cur_beta = Dequantize(beta_data[i],
                                layer_norm_bias_scale,
                                layer_norm_bias_zero_point);
        output[i] = output[i] / e * cur_gamma + cur_beta;
      }
    }, 0);

    if (failed.load(std::memory_order_acquire)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "input index out of range");
    }
  }

  // Calculate mask
  if (nullptr != inputs.mask) {
    const int32_t* mask_data = inputs.mask->template Data<int32_t>();
    for (int b = 0; b < batch_size; b++) {
      // TODO(kreeger): Fix static cast warning here:
      mask_index->template MutableData<int32_t>()[b] =
          static_cast<int32_t>(std::count_if(mask_data + (b * sequence_length),
                                             mask_data + (b * sequence_length) + sequence_length,
                                             [](int v) { return v == 1; }));
    }
  } else {
    memset(mask_index->template MutableData<int32_t>(), 0, batch_size * sizeof(int32_t));
  }
  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
