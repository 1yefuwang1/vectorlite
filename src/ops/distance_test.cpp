#include "distance.h"

#include <random>

#include "gtest/gtest.h"

static std::vector<float> GenerateOneRandomVector(size_t dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10000.0f, 10000.0f);

  std::vector<float> vec;
  vec.reserve(dim);
  for (int j = 0; j < dim; ++j) {
    vec.push_back(dis(gen));
  }

  return vec;
}

static std::vector<std::vector<float>> GenerateRandomVectors(size_t num_vectors, size_t dim) {
  static std::vector<std::vector<float>> data;
  if (data.size() == num_vectors) {
    return data;
  }

  data.reserve(num_vectors);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-10000.0f, 10000.0f);

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

TEST(InnerProduct, ShouldWorkWithRandomVectors) {
  std::cout << "Selected target: " << vectorlite::distance::DetectTarget() << std::endl;
  for (int dim = 4; dim <= 4; dim++) {
    auto vectors = GenerateRandomVectors(100, dim);
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

