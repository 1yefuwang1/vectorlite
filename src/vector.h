#pragma once

#include <string_view>

#include <rapidjson/document.h>
#include <hnswlib/hnswlib.h>

namespace sqlite_vector {

class Vector {
 public:
	Vector() = delete;

	static Vector FromJSON(std::string_view json) {
		rapidjson::Document doc;
		doc.Parse(json.data(), json.size());
		auto err = doc.GetParseError();
		if (err != rapidjson::ParseErrorCode::kParseErrorNone) {
			throw std::runtime_error("Failed to parse JSON");
		}
	}

 private:

};

} // end namespace sqlite_hnsw