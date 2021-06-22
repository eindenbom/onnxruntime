// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "embed_layer_norm.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

// Quantized version of QEmbedLayerNorm.
template <typename T>
class QEmbedLayerNorm final : public EmbedLayerNorm<T> {
 public:
  explicit QEmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

private:
  // Ergonomic class for holding all the various inputs for QEmbedLayerNorm:
 class QInputs : public Inputs {
  public:
   explicit QInputs(const Tensor* input_ids,
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
                    const Tensor* mask);

    const Tensor* word_embedding_scale;
    const Tensor* position_embedding_scale;
    const Tensor* segment_embedding_scale;
    const Tensor* gamma_scale;
    const Tensor* beta_scale;
    const Tensor* word_embedding_zero_point;
    const Tensor* position_embedding_zero_point;
    const Tensor* segment_embedding_zero_point;
    const Tensor* gamma_zero_point;
    const Tensor* beta_zero_point;
  };

  Status CheckInputs(const QInputs& inputs) const;
  Status ComputeInternal(OpKernelContext* context, const QInputs& inputs) const;
};

}  // namespace contrib
}  // namespace onnxruntime
