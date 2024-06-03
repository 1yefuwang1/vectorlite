#include "vector.h"

#include <iostream>

#include "gtest/gtest.h"
#include "vector_space.h"

TEST(VectorTest, FromJSON) {
  // Test valid JSON input
  std::string json = "[1.0, 2.0, 3.0]";
  auto result = vectorlite::Vector::FromJSON(json);
  EXPECT_TRUE(result.ok());
  vectorlite::Vector v = result.value();
  EXPECT_EQ(v.data().size(), 3);
  EXPECT_FLOAT_EQ(v.data()[0], 1.0);
  EXPECT_FLOAT_EQ(v.data()[1], 2.0);
  EXPECT_FLOAT_EQ(v.data()[2], 3.0);

  // Test invalid JSON type
  json = R"({"data": "invalid"})";
  result = vectorlite::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());

  // Test invalid array element
  json = R"([1.0, 2.0, "invalid"])";
  result = vectorlite::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());

  // Test invalid JSON
  json = R"(abc)";
  result = vectorlite::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());
}

TEST(VectorTest, Reversible_ToJSON_FromJSON) {
  // Test empty vector
  vectorlite::Vector v;
  std::string json = v.ToJSON();
  EXPECT_EQ(json, R"([])");

  // Test non-empty vector
  vectorlite::Vector v1({1.0, 2.0, 3.0});
  json = v1.ToJSON();
  auto parse_result = vectorlite::Vector::FromJSON(json);
  EXPECT_TRUE(parse_result.ok());
  const auto& parsed = *parse_result;
  EXPECT_EQ(parsed.data().size(), 3);
  EXPECT_FLOAT_EQ(parsed.data()[0], v1.data()[0]);
  EXPECT_FLOAT_EQ(parsed.data()[1], v1.data()[1]);
  EXPECT_FLOAT_EQ(parsed.data()[2], v1.data()[2]);
}

TEST(VectorTest, Reversible_ToBinary_FromBinary) {
  std::vector<float> data = {1.1, 2.23, 3.0};

  vectorlite::Vector v1(data);

  auto v2 = vectorlite::Vector::FromBlob(v1.ToBlob());
  EXPECT_TRUE(v2.ok());
  EXPECT_EQ(v1.data(), v2->data());
}

TEST(VectorTest, FromBinaryShouldFailWithInvalidInput) {
  auto v1 = vectorlite::Vector::FromBlob(std::string_view("aaa"));
  EXPECT_FALSE(v1.ok());
}

TEST(VectorDistance, ShouldWork) {
  // Test valid input
  vectorlite::Vector v1({1.0, 2.0, 3.0});
  vectorlite::Vector v2({4.0, 5.0, 6.0});
  auto distance = Distance(v1, v2, vectorlite::SpaceType::L2);
  EXPECT_TRUE(distance.ok());
  EXPECT_FLOAT_EQ(*distance, 27);

  distance = Distance(v2, v1, vectorlite::SpaceType::InnerProduct);
  EXPECT_TRUE(distance.ok());
  EXPECT_FLOAT_EQ(*distance, -31);

  distance = Distance(v1, v2, vectorlite::SpaceType::Cosine);
  EXPECT_TRUE(distance.ok());
  EXPECT_FLOAT_EQ(*distance, 0.025368214);

  // Test 0 dimension
  vectorlite::Vector v3;
  vectorlite::Vector v4;
  for (auto space :
       {vectorlite::SpaceType::L2, vectorlite::SpaceType::InnerProduct}) {
    distance = Distance(v3, v4, space);
    EXPECT_FALSE(distance.ok());
  }
}

TEST(VectorTest, Normalize) {
  vectorlite::Vector v({1.0, 2.0, 3.0});
  auto normalized = v.Normalize();
  EXPECT_FLOAT_EQ(normalized.data()[0], 0.26726124);
  EXPECT_FLOAT_EQ(normalized.data()[1], 0.53452247);
  EXPECT_FLOAT_EQ(normalized.data()[2], 0.8017837);
}
