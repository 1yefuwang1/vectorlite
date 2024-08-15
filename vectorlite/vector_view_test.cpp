#include "vector_view.h"

#include <string_view>

#include "gtest/gtest.h"
#include "vector.h"
#include "vector_space.h"

TEST(VectorViewTest, Reversible_ToBinary_FromBinary) {
  std::vector<float> data = {1.1, 2.23, 3.0};

  std::string_view blob(reinterpret_cast<const char*>(data.data()),
                        data.size() * sizeof(float));

  auto v2 = vectorlite::VectorView::FromBlob(blob);
  EXPECT_TRUE(v2.ok());
  EXPECT_EQ(data.size(), v2->dim());
  EXPECT_EQ(blob, v2->ToBlob());
}

TEST(VectorViewTest, FromBinaryShouldFailWithInvalidInput) {
  auto v1 = vectorlite::VectorView::FromBlob(std::string_view("aaa"));
  EXPECT_FALSE(v1.ok());
}

TEST(VectorViewTest, ToJSON) {
  std::vector<float> data = {1.1, 2.23, 3.0};
  vectorlite::VectorView v(data);

  std::string json = v.ToJSON();
  auto result = vectorlite::Vector::FromJSON(json);
  EXPECT_TRUE(result.ok());

  for (int i = 0; i < data.size(); i++) {
    EXPECT_FLOAT_EQ(data[i], result->data()[i]);
  }
}
