#pragma once
// Pure C++ (no HIP). Implemented by [LLM-IO] in src/io/safetensors.cpp, consumed by [LLM-CORE].
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace hipgpt2 {

struct TensorView {
    std::string          name;
    std::string          dtype;   // safetensors dtype string, e.g. "F32"
    std::vector<int64_t> shape;
    const uint8_t*       data;    // valid while the owning SafeTensors (or a copy) is alive
    size_t               nbytes;
    int64_t numel() const;        // product of shape (1 for scalars)
};

// Read-only safetensors file. Copies share one immutable buffer (cheap to copy).
class SafeTensors {
public:
    // Throws std::runtime_error (message includes the path) on I/O or format errors:
    // truncated file, header length > file size, bad JSON, data_offsets out of range,
    // nbytes != numel * sizeof(dtype).
    static SafeTensors open(const std::string& path);

    bool                     has(const std::string& name) const;
    const TensorView&        get(const std::string& name) const;   // throws if missing
    std::vector<std::string> names() const;                        // sorted, excludes "__metadata__"

    // Typed access. Throws std::runtime_error naming the tensor if it is missing, its dtype
    // is not "F32", or its shape != expect_shape. The pointer is aligned to alignof(float),
    // little-endian host assumed.
    const float* f32(const std::string& name, std::initializer_list<int64_t> expect_shape) const;

private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

}  // namespace hipgpt2
