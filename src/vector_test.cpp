#include "vector.h"
#include "gtest/gtest.h"
#include "hnswlib/hnswlib.h"

TEST(VectorTest, FromJSON) {
  // Test valid JSON input
  std::string json = "[1.0, 2.0, 3.0]";
  sqlite_vector::Vector v;
  auto result = sqlite_vector::Vector::FromJSON(json, &v);
  EXPECT_EQ(result, sqlite_vector::Vector::ParseResult::kOk);
  EXPECT_EQ(v.get_data().size(), 3);
  EXPECT_FLOAT_EQ(v.get_data()[0], 1.0);
  EXPECT_FLOAT_EQ(v.get_data()[1], 2.0);
  EXPECT_FLOAT_EQ(v.get_data()[2], 3.0);

  // Test invalid JSON type
  json = R"({"data": "invalid"})";
  result = sqlite_vector::Vector::FromJSON(json, &v);
  EXPECT_EQ(result, sqlite_vector::Vector::ParseResult::kInvalidJSONType);

  // Test invalid array element
  json = R"([1.0, 2.0, "invalid"])";
  result = sqlite_vector::Vector::FromJSON(json, &v);
  EXPECT_EQ(result, sqlite_vector::Vector::ParseResult::kInvalidElementType);

  // Test invalid JSON
  json = R"(abc)";
  result = sqlite_vector::Vector::FromJSON(json, &v);
  EXPECT_EQ(result, sqlite_vector::Vector::ParseResult::kParseFailed);
}

TEST(VectorTest, ToJSON) {
  // Test empty vector
  sqlite_vector::Vector v;
  std::string json = v.ToJSON();
  EXPECT_EQ(json, R"([])");

  // Test non-empty vector
  sqlite_vector::Vector v1({1.0, 2.0, 3.0});
  json = v1.ToJSON();
  auto parse_result = sqlite_vector::Vector::FromJSON(json, &v);
  EXPECT_EQ(parse_result, sqlite_vector::Vector::ParseResult::kOk);
  EXPECT_EQ(v.get_data().size(), 3);
  EXPECT_FLOAT_EQ(v.get_data()[0], 1.0);
  EXPECT_FLOAT_EQ(v.get_data()[1], 2.0);
  EXPECT_FLOAT_EQ(v.get_data()[2], 3.0);
}

TEST(VectorDistance, L2) {
  // Test valid input
  sqlite_vector::Vector v1({1.0, 2.0, 3.0});
  sqlite_vector::Vector v2({4.0, 5.0, 6.0});
  float distance = L2Distance(v1, v2);
  EXPECT_FLOAT_EQ(distance, 27);

  // Test empty input
  sqlite_vector::Vector v3;
  sqlite_vector::Vector v4;
  distance = L2Distance(v3, v4);
  EXPECT_FLOAT_EQ(distance, 0);

}
