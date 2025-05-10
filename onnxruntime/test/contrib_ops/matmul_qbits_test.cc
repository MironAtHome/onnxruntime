// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include <optional>
#include <string>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/narrow.h"
#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/qmath.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {

namespace test {

namespace {

const char* type_name_pairs[][2] = {
/*GGML_TYPE_Q4_0*/ {"q8_0", "q4_0"},
/*GGML_TYPE_Q4_1*/ {"q8_1", "q4_1"},
/*GGML_TYPE_Q5_0*/ {"q8_0", "q5_0"},
/*GGML_TYPE_Q5_1*/ {"q8_1", "q5_1"},
/*GGML_TYPE_Q8_0*/ {"q8_0", "q8_0"},
///*GGML_TYPE_Q8_1*/ {"q8_1", "q8_1"}, // no vec_dot
/*GGML_TYPE_Q2_K*/ {"q8_K", "q2_K"},
/*GGML_TYPE_Q3_K*/ {"q8_K", "q3_K"},
/*GGML_TYPE_Q4_K*/ {"q8_K", "q4_K"},
/*GGML_TYPE_Q5_K*/ {"q8_K", "q5_K"},
/*GGML_TYPE_Q6_K*/ {"q8_K", "q6_K"},
// /*GGML_TYPE_IQ2_XXS*/ {"q8_K", "iq2_xxs"}, // no quantization function
// /*GGML_TYPE_IQ2_XS*/ {"q8_K", "iq2_xs"},
// /*GGML_TYPE_IQ3_XXS*/ {"q8_K", "iq3_xxs"},
// /*GGML_TYPE_IQ3_S*/ {"q8_K", "iq3_s"},
// /*GGML_TYPE_IQ2_S*/ {"q8_K", "iq2_s"},
// /*GGML_TYPE_IQ1_S*/ {"q8_K", "iq1_s"},
// /*GGML_TYPE_IQ1_M*/ {"q8_K", "iq1_m"},
///////*GGML_TYPE_IQ4_NL*/ {"q8_0", "iq4_nl"},
///////*GGML_TYPE_IQ4_XS*/ {"q8_K", "iq4_xs"},
// /*GGML_TYPE_Q8_K*/ {"q8_K", "iq4_xs"},
/*GGML_TYPE_TQ1_0*/ {"q8_K", "tq1_0"},
/*GGML_TYPE_TQ2_0*/ {"q8_K", "tq2_0"},
/*GGML_TYPE_I2_S*/ {"i8_s", "i2_s"},
};

static std::string get_a_quant_type_name_for_b_quant(const std::string& b_quant_type_name) {
  const size_t num_pairs = sizeof(type_name_pairs) / sizeof(type_name_pairs[0]);
  for (size_t i = 0; i < num_pairs; i++) {
    if (type_name_pairs[i][1] == b_quant_type_name) {
      return type_name_pairs[i][0];
    }
  }
  throw std::runtime_error("wrong b_quant_type: " + b_quant_type_name);
}

void QuantizeDequantize(std::vector<float>& raw_vals,
                        std::vector<uint8_t>& quant_vals,
                        int32_t N,
                        int32_t K,
                        const std::string& quant_type_name) {
  auto& ortenv = **ort_env.get();
  onnxruntime::concurrency::ThreadPool* tp = ortenv.GetEnvironment().GetIntraOpThreadPool();

  size_t quant_size = MlasLowBitQuantizeSizeInByte(N, K, quant_type_name);
  quant_vals.resize(quant_size);

  MlasLowBitQuantize(&raw_vals[0], N, K, quant_type_name, &quant_vals[0], tp);

  size_t dequant_size = MlasLowBitDequantizeDataCount(N, K, quant_type_name);
  raw_vals.resize(dequant_size);

  MlasLowBitDequantize(&quant_vals[0], N, K, quant_type_name, &raw_vals[0], tp);
}

struct TestOptions {
  int64_t M{1};
  int64_t N{1};
  int64_t K{1};
  bool has_bias{false};
  std::string b_quant_type_name;

  std::optional<float> output_abs_error{};
  std::optional<float> output_rel_error{};
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& os, const TestOptions& opts) {
  return os << "M:" << opts.M << ", N:" << opts.N << ", K:" << opts.K
            << ", has_bias:" << opts.has_bias;
}

void RunTest(const TestOptions& opts,
             std::vector<std::unique_ptr<IExecutionProvider>>&& explicit_eps = {}) {
  SCOPED_TRACE(opts);

  const int64_t M = opts.M,
                K = opts.K,
                N = opts.N;
  const std::string& b_quant_type_name = opts.b_quant_type_name;
  std::string a_quant_type_name = get_a_quant_type_name_for_b_quant(b_quant_type_name);
  bool requantize_a =
      MlasLowBitCanQuantize(a_quant_type_name) && MlasLowBitCanDequantize(a_quant_type_name);

  // without being able to requantize, have reduce activation range to pass accuracy test
  RandomValueGenerator random{1234};
  std::vector<float> input0_vals(random.Gaussian<float>(AsSpan({M, K}), 0.0f, requantize_a ? 25.0f : 0.25f));
  std::vector<float> input1_f_vals(random.Gaussian<float>(AsSpan({K, N}), 0.0f, requantize_a ? 25.0f : 0.25f));

  std::vector<uint8_t> input1_vals;

  QuantizeDequantize(input1_f_vals,
                     input1_vals,
                     static_cast<int32_t>(N),
                     static_cast<int32_t>(K),
                     b_quant_type_name);

  if (requantize_a)
  {
    std::vector<uint8_t> input0_vals_dummy;
    QuantizeDequantize(input0_vals,
                       input0_vals_dummy,
                       static_cast<int32_t>(M),
                       static_cast<int32_t>(K),
                       a_quant_type_name);
  }
  const std::vector<int64_t> bias_shape = {N};
  const auto bias = [&]() -> std::optional<std::vector<float>> {
    if (opts.has_bias) {
      return random.Uniform(bias_shape, 1.0f, 5.0f);
    }
    return std::nullopt;
  }();

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        // weights are columnwise
        sum += input0_vals[m * K + k] * input1_f_vals[n * K + k];
      }
      expected_vals[m * N + n] = sum + (bias.has_value() ? (*bias)[n] : 0.0f);
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", K);
  test.AddAttribute<int64_t>("N", N);
  test.AddAttribute<std::string>("b_quant_type_name", b_quant_type_name);

  test.AddInput<float>("A", {M, K}, input0_vals, false);

  test.AddInput<uint8_t>("B", {static_cast<int64_t>(input1_vals.size())}, input1_vals, true);

  if (bias.has_value()) {
    test.AddInput<float>("bias", bias_shape, *bias, true);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  if (opts.output_abs_error.has_value()) {
    test.SetOutputAbsErr("Y", *opts.output_abs_error);
  }

  if (opts.output_rel_error.has_value()) {
    test.SetOutputRelErr("Y", *opts.output_rel_error);
  }

  if (!explicit_eps.empty()) {
    test.ConfigEps(std::move(explicit_eps));
  }

  test.RunWithConfig();
}

}  // namespace

template <int M, int N, int K, const char* b_quant_type_name>
void TestMatMulQBitsTyped() {
  TestOptions base_opts{};
  base_opts.M = M, base_opts.N = N, base_opts.K = K;

  base_opts.output_abs_error = 0.1f;
  base_opts.output_rel_error = 0.02f;

  {
    TestOptions opts = base_opts;
    opts.b_quant_type_name = b_quant_type_name;  // Assign the quantization type name

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.emplace_back(DefaultCpuExecutionProvider());
    RunTest(opts, std::move(execution_providers));
  }
}

constexpr const char q4_0[] = "q4_0";
TEST(MatMulQBits, q4_0) {
  TestMatMulQBitsTyped<1, 1, 256, q4_0>();
  TestMatMulQBitsTyped<1, 2, 256, q4_0>();
  TestMatMulQBitsTyped<2, 1, 256, q4_0>();
  TestMatMulQBitsTyped<2, 2, 256, q4_0>();
}

constexpr const char q4_1[] = "q4_1";
TEST(MatMulQBits, q4_1) {
  TestMatMulQBitsTyped<1, 1, 256, q4_1>();
  TestMatMulQBitsTyped<1, 2, 256, q4_1>();
  TestMatMulQBitsTyped<2, 1, 256, q4_1>();
  TestMatMulQBitsTyped<2, 2, 256, q4_1>();
}

constexpr const char q5_0[] = "q5_0";
TEST(MatMulQBits, q5_0) {
  TestMatMulQBitsTyped<1, 1, 256, q5_0>();
  TestMatMulQBitsTyped<1, 2, 256, q5_0>();
  TestMatMulQBitsTyped<2, 1, 256, q5_0>();
  TestMatMulQBitsTyped<2, 2, 256, q5_0>();
}

constexpr const char q5_1[] = "q5_1";
TEST(MatMulQBits, q5_1) {
  TestMatMulQBitsTyped<1, 1, 256, q5_1>();
  TestMatMulQBitsTyped<1, 2, 256, q5_1>();
  TestMatMulQBitsTyped<2, 1, 256, q5_1>();
  TestMatMulQBitsTyped<2, 2, 256, q5_1>();
}

constexpr const char q8_0[] = "q8_0";
TEST(MatMulQBits, q8_0) {
  TestMatMulQBitsTyped<1, 1, 256, q8_0>();
  TestMatMulQBitsTyped<1, 2, 256, q8_0>();
  TestMatMulQBitsTyped<2, 1, 256, q8_0>();
  TestMatMulQBitsTyped<2, 2, 256, q8_0>();
}

constexpr const char q2_K[] = "q2_K";
TEST(MatMulQBits, q2_K) {
  TestMatMulQBitsTyped<1, 1, 256, q2_K>();
  TestMatMulQBitsTyped<1, 2, 256, q2_K>();
  TestMatMulQBitsTyped<2, 1, 256, q2_K>();
  TestMatMulQBitsTyped<2, 2, 256, q2_K>();
}

constexpr const char q3_K[] = "q3_K";
TEST(MatMulQBits, q3_K) {
  TestMatMulQBitsTyped<1, 1, 256, q3_K>();
  TestMatMulQBitsTyped<1, 2, 256, q3_K>();
  TestMatMulQBitsTyped<2, 1, 256, q3_K>();
  TestMatMulQBitsTyped<2, 2, 256, q3_K>();
}

constexpr const char q4_K[] = "q4_K";
TEST(MatMulQBits, q4_K) {
  TestMatMulQBitsTyped<1, 1, 256, q4_K>();
  TestMatMulQBitsTyped<1, 2, 256, q4_K>();
  TestMatMulQBitsTyped<2, 1, 256, q4_K>();
  TestMatMulQBitsTyped<2, 2, 256, q4_K>();
}

constexpr const char q5_K[] = "q5_K";
TEST(MatMulQBits, q5_K) {
  TestMatMulQBitsTyped<1, 1, 256, q5_K>();
  TestMatMulQBitsTyped<1, 2, 256, q5_K>();
  TestMatMulQBitsTyped<2, 1, 256, q5_K>();
  TestMatMulQBitsTyped<2, 2, 256, q5_K>();
}

constexpr const char q6_K[] = "q6_K";
TEST(MatMulQBits, q6_K) {
  TestMatMulQBitsTyped<1, 1, 256, q6_K>();
  TestMatMulQBitsTyped<1, 2, 256, q6_K>();
  TestMatMulQBitsTyped<2, 1, 256, q6_K>();
  TestMatMulQBitsTyped<2, 2, 256, q6_K>();
}

constexpr const char tq1_0[] = "tq1_0";
TEST(MatMulQBits, tq1_0) {
  TestMatMulQBitsTyped<1, 1, 256, tq1_0>();
  TestMatMulQBitsTyped<1, 2, 256, tq1_0>();
  TestMatMulQBitsTyped<2, 1, 256, tq1_0>();
  TestMatMulQBitsTyped<2, 2, 256, tq1_0>();
}

constexpr const char tq2_0[] = "tq2_0";
TEST(MatMulQBits, tq2_0) {
  TestMatMulQBitsTyped<1, 1, 256, tq2_0>();
  TestMatMulQBitsTyped<1, 2, 256, tq2_0>();
  TestMatMulQBitsTyped<2, 1, 256, tq2_0>();
  TestMatMulQBitsTyped<2, 2, 256, tq2_0>();
}

constexpr const char i2_s[] = "i2_s";
TEST(MatMulQBits, i2_s) {
  TestMatMulQBitsTyped<1, 1, 256, i2_s>();
  TestMatMulQBitsTyped<1, 2, 256, i2_s>();
  TestMatMulQBitsTyped<2, 1, 256, i2_s>();
  TestMatMulQBitsTyped<2, 2, 256, i2_s>();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
