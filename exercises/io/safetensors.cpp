#include "safetensors.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>

#include "json_mini.hpp"

namespace hipgpt2 {

int64_t TensorView::numel() const {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    return n;
}

namespace {
struct AlignedFree {
    void operator()(uint8_t* p) const { std::free(p); }
};

size_t dtype_size(const std::string& d) {
    if (d == "F64" || d == "I64" || d == "U64") return 8;
    if (d == "F32" || d == "I32" || d == "U32") return 4;
    if (d == "F16" || d == "BF16" || d == "I16" || d == "U16") return 2;
    if (d == "I8" || d == "U8" || d == "BOOL" || d == "F8_E4M3" || d == "F8_E5M2") return 1;
    return 0;
}
}  // namespace

struct SafeTensors::Impl {
    std::string path;
    std::unique_ptr<uint8_t, AlignedFree> buf;  // data section, 64-byte aligned
    size_t buf_size = 0;
    std::map<std::string, TensorView> tensors;
};

SafeTensors SafeTensors::open(const std::string& path) {
    auto fail = [&](const std::string& m) { throw std::runtime_error(path + ": " + m); };
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) fail("cannot open file");
    struct Closer { FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    if (std::fseek(f, 0, SEEK_END) != 0) fail("seek failed");
    long fsz_l = std::ftell(f);
    if (fsz_l < 0) fail("cannot determine file size");
    uint64_t fsz = (uint64_t)fsz_l;
    std::rewind(f);
    if (fsz < 8) fail("truncated file (shorter than 8-byte header length)");
    uint8_t lenb[8];
    if (std::fread(lenb, 1, 8, f) != 8) fail("read error");
    uint64_t n = 0;
    for (int k = 7; k >= 0; --k) n = (n << 8) | lenb[k];
    if (n > fsz - 8) fail("header length " + std::to_string(n) + " exceeds file size");
    std::string header(n, '\0');
    if (n && std::fread(&header[0], 1, n, f) != n) fail("read error in header");
    json::Value root;
    try {
        root = json::parse(header);
    } catch (const std::exception& e) {
        fail(std::string("bad header JSON: ") + e.what());
    }
    if (root.type != json::Value::Obj) fail("header JSON is not an object");

    auto impl = std::make_shared<Impl>();
    impl->path = path;
    uint64_t data_size = fsz - 8 - n;
    impl->buf_size = (size_t)data_size;
    size_t alloc = ((size_t)data_size + 63) / 64 * 64;
    if (alloc == 0) alloc = 64;
    void* mem = std::aligned_alloc(64, alloc);
    if (!mem) throw std::bad_alloc();
    impl->buf.reset(static_cast<uint8_t*>(mem));
    if (data_size && std::fread(impl->buf.get(), 1, (size_t)data_size, f) != data_size) fail("read error in data section");

    for (const auto& kv : root.obj) {
        const std::string& name = kv.first;
        if (name == "__metadata__") continue;
        const json::Value& e = kv.second;
        if (e.type != json::Value::Obj) fail("tensor '" + name + "': entry is not an object");
        const json::Value* dt = e.find("dtype");
        const json::Value* sh = e.find("shape");
        const json::Value* off = e.find("data_offsets");
        if (!dt || dt->type != json::Value::Str) fail("tensor '" + name + "': missing dtype");
        if (!sh || sh->type != json::Value::Arr) fail("tensor '" + name + "': missing shape");
        if (!off || off->type != json::Value::Arr || off->arr.size() != 2 || !off->arr[0].is_int() ||
            !off->arr[1].is_int())
            fail("tensor '" + name + "': bad data_offsets");
        TensorView tv;
        tv.name = name;
        tv.dtype = dt->s;
        int64_t numel = 1;
        for (const auto& d : sh->arr) {
            if (!d.is_int() || d.i < 0) fail("tensor '" + name + "': bad shape entry");
            tv.shape.push_back(d.i);
            if (d.i != 0 && numel > INT64_MAX / d.i) fail("tensor '" + name + "': shape overflow");
            numel *= d.i;
        }
        int64_t b = off->arr[0].i, en = off->arr[1].i;
        if (b < 0 || en < b || (uint64_t)en > data_size)
            fail("tensor '" + name + "': data_offsets [" + std::to_string(b) + ", " + std::to_string(en) +
                 "] out of range (data section is " + std::to_string(data_size) + " bytes)");
        size_t es = dtype_size(tv.dtype);
        if (es == 0) fail("tensor '" + name + "': unsupported dtype " + tv.dtype);
        if ((uint64_t)(en - b) != (uint64_t)numel * es)
            fail("tensor '" + name + "': nbytes " + std::to_string(en - b) + " != numel*sizeof(dtype) " +
                 std::to_string((uint64_t)numel * es));
        tv.nbytes = (size_t)(en - b);
        tv.data = impl->buf.get() + b;
        impl->tensors.emplace(name, std::move(tv));
    }
    SafeTensors st;
    st.impl_ = std::move(impl);
    return st;
}

bool SafeTensors::has(const std::string& name) const { return impl_->tensors.count(name) != 0; }

const TensorView& SafeTensors::get(const std::string& name) const {
    auto it = impl_->tensors.find(name);
    if (it == impl_->tensors.end())
        throw std::runtime_error(impl_->path + ": tensor '" + name + "' not found");
    return it->second;
}

std::vector<std::string> SafeTensors::names() const {
    std::vector<std::string> v;
    for (const auto& kv : impl_->tensors) v.push_back(kv.first);  // std::map => sorted
    return v;
}

const float* SafeTensors::f32(const std::string& name, std::initializer_list<int64_t> expect_shape) const {
    auto it = impl_->tensors.find(name);
    if (it == impl_->tensors.end())
        throw std::runtime_error(impl_->path + ": tensor '" + name + "' not found");
    const TensorView& t = it->second;
    if (t.dtype != "F32")
        throw std::runtime_error(impl_->path + ": tensor '" + name + "' has dtype " + t.dtype + ", expected F32");
    std::vector<int64_t> want(expect_shape);
    if (t.shape != want) {
        auto fmt = [](const std::vector<int64_t>& s) {
            std::string r = "[";
            for (size_t k = 0; k < s.size(); ++k) r += (k ? ", " : "") + std::to_string(s[k]);
            return r + "]";
        };
        throw std::runtime_error(impl_->path + ": tensor '" + name + "' has shape " + fmt(t.shape) +
                                 ", expected " + fmt(want));
    }
    // Data offset is not necessarily 4-aligned in a foreign file; the buffer base is 64-aligned and
    // the header size is typically padded to 8 by writers. Enforce the documented guarantee.
    if (reinterpret_cast<uintptr_t>(t.data) % alignof(float) != 0)
        throw std::runtime_error(impl_->path + ": tensor '" + name + "' data is not 4-byte aligned");
    return reinterpret_cast<const float*>(t.data);
}

}  // namespace hipgpt2
