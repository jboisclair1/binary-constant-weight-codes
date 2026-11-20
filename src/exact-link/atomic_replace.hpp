#pragma once
// Atomic replacement of an existing checkpoint on POSIX and Windows.
#include <filesystem>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
inline void replace_checkpoint(const std::filesystem::path& source,
                               const std::filesystem::path& destination) {
#ifdef _WIN32
    if (!MoveFileExW(source.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("Could not replace the checkpoint file.");
    }
#else
    std::filesystem::rename(source, destination);
#endif
}
