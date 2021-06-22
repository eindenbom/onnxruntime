// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
class EmbedLayerNorm : public OpKernel {
 public:
  explicit EmbedLayerNorm(const OpKernelInfo& op_kernel_info);
  Status Compute(OpKernelContext* context) const override;

 protected:
  // Ergonomic class for holding all the various inputs for EmbedLayerNorm:
  class Inputs {
   public:
    explicit Inputs(const Tensor* input_ids,
                    const Tensor* segment_ids,
                    const Tensor* word_embedding,
                    const Tensor* position_embedding,
                    const Tensor* segment_embedding,
                    const Tensor* gamma,
                    const Tensor* beta,
                    const Tensor* mask);

    const Tensor* input_ids;
    const Tensor* segment_ids;
    const Tensor* word_embedding;
    const Tensor* position_embedding;
    const Tensor* segment_embedding;
    const Tensor* gamma;
    const Tensor* beta;
    const Tensor* mask;

    int batch_size;
    int sequence_size;
    int hidden_size;
    bool has_segment_embedding;
    int word_embedding_length;
    int position_embedding_length;
    int segment_embedding_length;
  };

  // TODO - docs.
  Status CheckInputs(const Inputs& inputs) const;
  Status ComputeInternal(OpKernelContext* context, const Inputs& inputs) const;

// TODO - this should be private.
  float epsilon_;
};

}  // namespace contrib
}  // namespace onnxruntime
