// Loads tests/data/golden.json (written by tests/gen_golden.py from the Python reference code).
#pragma once
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

#include "config.h"

inline const nlohmann::json& golden() {
    static const nlohmann::json j = [] {
        std::ifstream f(std::filesystem::path(find_root("")) / "server" / "cpp" / "tests" / "data" / "golden.json");
        if (!f) throw std::runtime_error("tests/data/golden.json missing: run server/cpp/tests/gen_golden.py");
        return nlohmann::json::parse(f);
    }();
    return j;
}
