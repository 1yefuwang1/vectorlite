#include "vector.h"

#include <iostream>

#include "gtest/gtest.h"

TEST(VectorTest, FromJSON) {
  // Test valid JSON input
  std::string json = "[1.0, 2.0, 3.0]";
  auto result = sqlite_vector::Vector::FromJSON(json);
  EXPECT_TRUE(result.ok());
  sqlite_vector::Vector v = result.value();
  EXPECT_EQ(v.data().size(), 3);
  EXPECT_FLOAT_EQ(v.data()[0], 1.0);
  EXPECT_FLOAT_EQ(v.data()[1], 2.0);
  EXPECT_FLOAT_EQ(v.data()[2], 3.0);

  // Test invalid JSON type
  json = R"({"data": "invalid"})";
  result = sqlite_vector::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());

  // Test invalid array element
  json = R"([1.0, 2.0, "invalid"])";
  result = sqlite_vector::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());

  // Test invalid JSON
  json = R"(abc)";
  result = sqlite_vector::Vector::FromJSON(json);
  EXPECT_FALSE(result.ok());
}

TEST(VectorTest, Reversible_ToJSON_FromJSON) {
  // Test empty vector
  sqlite_vector::Vector v;
  std::string json = v.ToJSON();
  EXPECT_EQ(json, R"([])");

  // Test non-empty vector
  sqlite_vector::Vector v1({1.0, 2.0, 3.0});
  json = v1.ToJSON();
  auto parse_result = sqlite_vector::Vector::FromJSON(json);
  EXPECT_TRUE(parse_result.ok());
  const auto& parsed = *parse_result;
  EXPECT_EQ(parsed.data().size(), 3);
  EXPECT_FLOAT_EQ(parsed.data()[0], v1.data()[0]);
  EXPECT_FLOAT_EQ(parsed.data()[1], v1.data()[1]);
  EXPECT_FLOAT_EQ(parsed.data()[2], v1.data()[2]);
}

TEST(VectorTest, MsgPack) {
  sqlite_vector::Vector v({1.01, 2.03, 3.01111});
  std::string msgpack = v.ToMsgPack();

  auto parse_result = sqlite_vector::Vector::FromMsgPack(msgpack);
  EXPECT_TRUE(parse_result.ok());
  const auto& parsed = *parse_result;
  EXPECT_EQ(parsed.data().size(), 3);
  EXPECT_FLOAT_EQ(parsed.data()[0], v.data()[0]);
  EXPECT_FLOAT_EQ(parsed.data()[1], v.data()[1]);
  EXPECT_FLOAT_EQ(parsed.data()[2], v.data()[2]);
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
