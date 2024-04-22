#include "util.h"

#include "gtest/gtest.h"

TEST(IsValidColumnNameTest, ValidColumnNames) {
  EXPECT_TRUE(vectorlite::IsValidColumnName("valid_column_name"));
  EXPECT_TRUE(vectorlite::IsValidColumnName("ValidColumnName"));
  EXPECT_TRUE(vectorlite::IsValidColumnName("_valid_column_name"));
  EXPECT_TRUE(vectorlite::IsValidColumnName("valid_column_name_1"));
  EXPECT_TRUE(vectorlite::IsValidColumnName("valid$column$name"));
}

TEST(IsValidColumnNameTest, InvalidColumnNames) {
  EXPECT_FALSE(vectorlite::IsValidColumnName(""));
  EXPECT_FALSE(vectorlite::IsValidColumnName("123"));
  EXPECT_FALSE(vectorlite::IsValidColumnName("invalid column name"));
  EXPECT_FALSE(vectorlite::IsValidColumnName("SELECT"));
  EXPECT_FALSE(vectorlite::IsValidColumnName("valid_column_name "));
}