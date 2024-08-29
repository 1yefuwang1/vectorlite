#include "ops.h"

#include <random>

#include "gtest/gtest.h"
#include "hnswlib/hnswlib.h"
#include "hwy/base.h"

static std::vector<std::vector<float>> GenerateRandomVectors(size_t num_vectors,
                                                             size_t dim) {
  std::vector<std::vector<float>> data;

  data.reserve(num_vectors);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0f, 1.0f);

  for (int i = 0; i < num_vectors; ++i) {
    std::vector<float> vec;
    vec.reserve(dim);
    for (int j = 0; j < dim; ++j) {
      vec.push_back(dis(gen));
    }
    data.push_back(vec);
  }

  return data;
}

static constexpr float kEpsilon = 1e-3;

TEST(InnerProduct, ShouldReturnZeroForEmptyVectors) {
  // Fixes C2466: cannot allocate an array of constant size 0 on MSVC
  float v1[] = {1};
  float v2[] = {1};
  auto result = vectorlite::ops::InnerProduct(v1, v2, 0);
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(InnerProduct, ShouldWorkWithRandomVectors) {
  for (int dim = 1; dim <= 1000; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    for (int i = 0; i < vectors.size(); ++i) {
      for (int j = 0; j < vectors.size(); ++j) {
        auto v1 = vectors[i].data();
        auto v2 = vectors[j].data();
        auto size = dim;
        auto result = vectorlite::ops::InnerProduct(v1, v2, size);
        float expected = 0;
        for (int k = 0; k < size; ++k) {
          expected += v1[k] * v2[k];
        }
        // Note: floating point operations are not associative. SIMD version and
        // scalar version traverse elements in different order. So the result
        // should be different but close enough
        EXPECT_NEAR(result, expected, kEpsilon);
      }
    }
  }
}

TEST(InnerProductDistance, ShouldReturnOneForEmptyVectors) {
  // Fixes C2466: cannot allocate an array of constant size 0 on MSVC
  float v1[] = {1};
  float v2[] = {1};
  auto result = vectorlite::ops::InnerProductDistance(v1, v2, 0);
  EXPECT_FLOAT_EQ(result, 1.0f);
}

TEST(InnerProductDistance, ShouldReturnSimilarResultToHNSWLIB) {
  for (size_t dim = 1; dim <= 1000; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    hnswlib::InnerProductSpace space(dim);
    auto dist_func = space.get_dist_func();
    for (int i = 0; i < vectors.size(); ++i) {
      for (int j = 0; j < vectors.size(); ++j) {
        auto v1 = vectors[i].data();
        auto v2 = vectors[j].data();
        float result = vectorlite::ops::InnerProductDistance(v1, v2, dim);
        // Note dim has to be of type size_t which is 64-bit
        // If you use int, it will be 32-bit and get a segmentation fault
        // because dist_func uses const void* to pass dim which erase the type
        // information.
        float hnswlib_result = dist_func(v1, v2, &dim);

        EXPECT_NEAR(result, hnswlib_result, 1e-3);
      }
    }
  }
}

TEST(L2DistanceSquared, ShouldReturnZeroForEmptyVectors) {
  // Fixes C2466: cannot allocate an array of constant size 0 on MSVC
  float v1[] = {1};
  float v2[] = {1};
  float result = vectorlite::ops::L2DistanceSquared(v1, v2, 0);
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(L2DistanceSquared, ShouldWorkWithRandomVectors) {
  for (size_t dim = 1; dim <= 1000; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    hnswlib::L2Space space(dim);
    auto dist_func = space.get_dist_func();
    for (int i = 0; i < vectors.size(); ++i) {
      for (int j = 0; j < vectors.size(); ++j) {
        auto v1 = vectors[i].data();
        auto v2 = vectors[j].data();
        float result = vectorlite::ops::L2DistanceSquared(v1, v2, dim);
        float hnswlib_result = dist_func(v1, v2, &dim);
        float expected = 0;
        for (int k = 0; k < dim; ++k) {
          float diff = v1[k] - v2[k];
          expected += diff * diff;
        }
        EXPECT_NEAR(result, hnswlib_result, 1e-2);
        EXPECT_NEAR(result, expected, 1e-2);
      }
    }
  }
}

TEST(Normalize, ShouldReturnCorrectResult) {
  for (int dim = 1; dim <= 1000; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    for (int i = 0; i < vectors.size(); ++i) {
      std::vector<float> v1 = vectors[i];
      std::vector<float> v2 = vectors[i];
      auto size = dim;
      vectorlite::ops::Normalize(v1.data(), size);

      vectorlite::ops::Normalize_Scalar(v2.data(), size);
      for (int j = 0; j < size; ++j) {
        EXPECT_NEAR(v1[j], v2[j], 1e-6);
      }
    }
  }
}

TEST(QuantizeF32ToBF16, ShouldReturnCorrectResult) {
  for (int dim = 0; dim <= 100; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    for (int i = 0; i < vectors.size(); ++i) {
      std::vector<float> v = vectors[i];
      auto size = dim;
      std::vector<hwy::bfloat16_t> out(size);
      vectorlite::ops::QuantizeF32ToBF16(v.data(), out.data(), size);

      for (int j = 0; j < size; ++j) {
        float expected = hwy::F32FromBF16(hwy::BF16FromF32(v[j]));
        EXPECT_NEAR(expected, hwy::F32FromBF16(out[j]), 1e-6)
            << "v[" << j << "] = " << v[j] << " dim = " << dim;
      }
    }
  }
}

TEST(QuantizeF32ToF16, ShouldReturnCorrectResult) {
  for (int dim = 0; dim <= 100; dim++) {
    auto vectors = GenerateRandomVectors(10, dim);
    for (int i = 0; i < vectors.size(); ++i) {
      std::vector<float> v = vectors[i];
      auto size = dim;
      std::vector<hwy::float16_t> out(size);
      vectorlite::ops::QuantizeF32ToF16(v.data(), out.data(), size);

      for (int j = 0; j < size; ++j) {
        float expected = hwy::F32FromF16(hwy::F16FromF32(v[j]));
        EXPECT_NEAR(expected, hwy::F32FromF16(out[j]), 1e-6)
            << "v[" << j << "] = " << v[j] << " dim = " << dim;
      }
    }
  }
}