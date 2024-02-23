#include "util.h"

#include "gtest/gtest.h"

TEST(IsValidColumnNameTest, ValidColumnNames) {
  EXPECT_TRUE(sqlite_vector::IsValidColumnName("valid_column_name"));
  EXPECT_TRUE(sqlite_vector::IsValidColumnName("ValidColumnName"));
  EXPECT_TRUE(sqlite_vector::IsValidColumnName("_valid_column_name"));
  EXPECT_TRUE(sqlite_vector::IsValidColumnName("valid_column_name_1"));
  EXPECT_TRUE(sqlite_vector::IsValidColumnName("valid$column$name"));
}

TEST(IsValidColumnNameTest, InvalidColumnNames) {
  EXPECT_FALSE(sqlite_vector::IsValidColumnName(""));
  EXPECT_FALSE(sqlite_vector::IsValidColumnName("123"));
  EXPECT_FALSE(sqlite_vector::IsValidColumnName("invalid column name"));
  EXPECT_FALSE(sqlite_vector::IsValidColumnName("SELECT"));
  EXPECT_FALSE(sqlite_vector::IsValidColumnName("valid_column_name "));
}