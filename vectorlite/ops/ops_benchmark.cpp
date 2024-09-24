#include <hwy/base.h>

#include <random>

#include "benchmark/benchmark.h"
#include "hnswlib/hnswlib.h"
#include "ops.h"

static std::vector<float> GenerateOneRandomVector(size_t dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1000.0f, 1000.0f);

  std::vector<float> vec;
  vec.reserve(dim);
  for (int j = 0; j < dim; ++j) {
    vec.push_back(dis(gen));
  }

  return vec;
}

static void BM_InnerProduct_Vectorlite(benchmark::State& state) {
  size_t dim = state.range(0);
  size_t self_product = state.range(1);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vectorlite::ops::InnerProductDistance(
        v1.data(), self_product ? v1.data() : v2.data(), dim));
    benchmark::ClobberMemory();
  }
}

static void BM_InnerProduct_Vectorlite_BF16(benchmark::State& state) {
  size_t dim = state.range(0);
  size_t self_product = state.range(1);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  std::vector<hwy::bfloat16_t> v1_bf16(dim);
  vectorlite::ops::QuantizeF32ToBF16(v1.data(), v1_bf16.data(), dim);

  std::vector<hwy::bfloat16_t> v2_bf16(dim);
  vectorlite::ops::QuantizeF32ToBF16(v2.data(), v2_bf16.data(), dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vectorlite::ops::InnerProductDistance(
        v1_bf16.data(), self_product ? v1_bf16.data() : v2_bf16.data(), dim));
    benchmark::ClobberMemory();
  }
}

static void BM_InnerProduct_Scalar(benchmark::State& state) {
  size_t dim = state.range(0);
  size_t self_product = state.range(1);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(hnswlib::InnerProductDistance(
        v1.data(), self_product ? v1.data() : v2.data(), &dim));
    benchmark::ClobberMemory();
  }
}

static void BM_InnerProduct_HNSWLIB(benchmark::State& state) {
  size_t dim = state.range(0);
  size_t self_product = state.range(1);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  hnswlib::InnerProductSpace space(dim);
  auto dist_func = space.get_dist_func();
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        dist_func(v1.data(), self_product ? v1.data() : v2.data(), &dim));
    benchmark::ClobberMemory();
  }
}

static void BM_L2DistanceSquared_Vectorlite(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        vectorlite::ops::L2DistanceSquared(v1.data(), v2.data(), dim));
    benchmark::ClobberMemory();
  }
}

static void BM_L2DistanceSquared_Vectorlite_BF16(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  std::vector<hwy::bfloat16_t> v1_bf16(dim);
  std::vector<hwy::bfloat16_t> v2_bf16(dim);
  vectorlite::ops::QuantizeF32ToBF16(v1.data(), v1_bf16.data(), dim);
  vectorlite::ops::QuantizeF32ToBF16(v2.data(), v2_bf16.data(), dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(vectorlite::ops::L2DistanceSquared(
        v1_bf16.data(), v2_bf16.data(), dim));
    benchmark::ClobberMemory();
  }
}

static void BM_L2DistanceSquared_Scalar(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    benchmark::DoNotOptimize(hnswlib::L2Sqr(v1.data(), v2.data(), &dim));
    benchmark::ClobberMemory();
  }
}

static void BM_L2DistanceSquared_HNSWLIB(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);
  auto v2 = GenerateOneRandomVector(dim);

  hnswlib::L2Space space(dim);
  auto dist_func = space.get_dist_func();
  for (auto _ : state) {
    benchmark::DoNotOptimize(dist_func(v1.data(), v2.data(), &dim));
    benchmark::ClobberMemory();
  }
}

static void BM_Normalize_Vectorlite(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    vectorlite::ops::Normalize(v1.data(), dim);
  }
}

static void BM_Normalize_Vectorlite_BF16(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);
  std::vector<hwy::bfloat16_t> v1_bf16(dim);
  vectorlite::ops::QuantizeF32ToBF16(v1.data(), v1_bf16.data(), dim);

  for (auto _ : state) {
    vectorlite::ops::Normalize(v1_bf16.data(), dim);
  }
}

static void BM_Normalize_Scalar(benchmark::State& state) {
  size_t dim = state.range(0);
  auto v1 = GenerateOneRandomVector(dim);

  for (auto _ : state) {
    vectorlite::ops::Normalize_Scalar(v1.data(), dim);
  }
}

BENCHMARK(BM_InnerProduct_Scalar)
    ->ArgsProduct({
        benchmark::CreateRange(128, 8 << 11, 2), {0, 1}  // self product
    });
BENCHMARK(BM_InnerProduct_HNSWLIB)
    ->ArgsProduct({
        benchmark::CreateRange(128, 8 << 11, 2), {0, 1}  // self product
    });
BENCHMARK(BM_InnerProduct_Vectorlite)
    ->ArgsProduct({
        benchmark::CreateRange(128, 8 << 11, 2), {0, 1}  // self product
    });
BENCHMARK(BM_InnerProduct_Vectorlite_BF16)
    ->ArgsProduct({benchmark::CreateRange(128, 8 << 11, 2), {0, 1}});
BENCHMARK(BM_Normalize_Vectorlite)->RangeMultiplier(2)->Range(128, 8 << 11);
BENCHMARK(BM_Normalize_Vectorlite_BF16)
    ->RangeMultiplier(2)
    ->Range(128, 8 << 11);
BENCHMARK(BM_Normalize_Scalar)->RangeMultiplier(2)->Range(128, 8 << 11);
BENCHMARK(BM_L2DistanceSquared_Scalar)->RangeMultiplier(2)->Range(128, 8 << 11);
BENCHMARK(BM_L2DistanceSquared_Vectorlite)
    ->RangeMultiplier(2)
    ->Range(128, 8 << 11);
BENCHMARK(BM_L2DistanceSquared_Vectorlite_BF16)
    ->RangeMultiplier(2)
    ->Range(128, 8 << 11);
BENCHMARK(BM_L2DistanceSquared_HNSWLIB)
    ->RangeMultiplier(2)
    ->Range(128, 8 << 11);