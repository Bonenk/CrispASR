#pragma once

#include <cmath>
#include <cstdio>
#include <vector>

inline bool dia_read_intervention_keys(const char* path, size_t count, std::vector<float>& values) {
    values.resize(count);
    FILE* file = std::fopen(path, "rb");
    if (!file)
        return false;
    const bool complete =
        std::fread(values.data(), sizeof(float), count, file) == count && std::fgetc(file) == EOF && !std::ferror(file);
    std::fclose(file);
    if (!complete)
        return false;
    for (float value : values)
        if (!std::isfinite(value))
            return false;
    return true;
}
