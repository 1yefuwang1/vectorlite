#include "distance.h"

#include <random>

#include "gtest/gtest.h"

static std::vector<std::vector<float>> GenerateRandomVectors(size_t num_vectors,
                                                             size_t dim) {
  static std::vector<std::vector<float>> data;
  if (data.size() == num_vectors) {
    return data;
  }

  data.reserve(num_vectors);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1000.0f, 1000.0f);

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

TEST(InnerProduct, ShouldReturnZeroForEmptyVectors) {
  float v1[] = {};
  float v2[] = {};
  auto result = vectorlite::distance::InnerProduct(v1, v2, 0);
  EXPECT_FLOAT_EQ(result, 0.0f);
}

TEST(InnerProduct, ShouldWorkWithRandomVectors) {
  for (int dim = 1; dim <= 10000; dim++) {
    auto vectors = GenerateRandomVectors(1, dim);
    for (int i = 0; i < vectors.size(); ++i) {
      for (int j = 0; j < vectors.size(); ++j) {
        auto v1 = vectors[i].data();
        auto v2 = vectors[j].data();
        auto size = dim;
        auto result = vectorlite::distance::InnerProduct(v1, v2, size);
        float expected = 0;
        for (int k = 0; k < size; ++k) {
          expected += v1[k] * v2[k];
        }
        EXPECT_FLOAT_EQ(result, expected);
      }
    }
  }
}
