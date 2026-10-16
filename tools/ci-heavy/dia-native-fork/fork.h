#pragma once
// Scratch-only observation/substitution after the unchanged nine-head sampler.
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct dia_native_fork {
    std::string dir;
    int replacement = -1;
    bool seen = false;

    bool configure(const char* mode, const char* directory) {
        if (!mode || !directory || !*directory)
            return false;
        dir = directory;
        if (std::strcmp(mode, "off") == 0)
            replacement = -1;
        else if (std::strcmp(mode, "identity") == 0)
            replacement = 904;
        else if (std::strcmp(mode, "restore") == 0)
            replacement = 877;
        else
            return false;
        return true;
    }

    bool write(const char* name, const void* data, size_t size) const {
        std::ofstream file(dir + "/" + name, std::ios::binary);
        if (!file)
            return false;
        file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        file.close();
        return bool(file);
    }

    static std::string state(const std::mt19937& rng) {
        std::ostringstream stream;
        stream << rng;
        return stream.str();
    }

    bool apply(uint32_t step, std::vector<uint32_t>& tokens, std::mt19937& rng) {
        if (step != 978)
            return true;
        if (seen || tokens.size() != 9 || tokens[2] != 904)
            return false;
        const auto original = tokens;
        const auto before = rng;
        if (replacement >= 0)
            tokens[2] = static_cast<uint32_t>(replacement);
        if (rng != before)
            return false;
        const auto first = state(before), last = state(rng);
        if (!write("sample-original.u32", original.data(), original.size() * sizeof(uint32_t)) ||
            !write("sample-used.u32", tokens.data(), tokens.size() * sizeof(uint32_t)) ||
            !write("rng-before.txt", first.data(), first.size()) || !write("rng-after.txt", last.data(), last.size()))
            return false;
        seen = true;
        return true;
    }

    bool finish(const std::vector<uint32_t>& delayed, const std::vector<uint32_t>& codes, const std::mt19937& rng) {
        if (!seen || delayed.empty() || codes.empty() || delayed.size() % 9 || codes.size() % 9)
            return false;
        const auto final = state(rng);
        return write("native-delayed.u32", delayed.data(), delayed.size() * sizeof(uint32_t)) &&
               write("native-codes.u32", codes.data(), codes.size() * sizeof(uint32_t)) &&
               write("rng-final.txt", final.data(), final.size());
    }
};
