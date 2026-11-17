#include "gpt2_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

#include "json_mini.hpp"

namespace hipgpt2 {

static int req_int(const json::Value& root, const char* key, const std::string& path, bool required,
                   int* out) {
    const json::Value* v = root.find(key);
    if (!v) {
        if (required) throw std::runtime_error(path + ": missing key '" + key + "'");
        return 0;
    }
    if (!v->is_int() || v->i <= 0 || v->i > 0x7fffffff)
        throw std::runtime_error(path + ": key '" + key + "' must be a positive integer");
    *out = (int)v->i;
    return 1;
}

Gpt2Config load_gpt2_config(const std::string& model_dir) {
    std::string path = model_dir + "/config.json";
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    json::Value root;
    try {
        root = json::parse(ss.str());
    } catch (const std::exception& e) {
        throw std::runtime_error(path + ": " + e.what());
    }
    if (root.type != json::Value::Obj) throw std::runtime_error(path + ": top level must be an object");
    Gpt2Config c;
    req_int(root, "n_layer", path, true, &c.n_layer);
    req_int(root, "n_head", path, true, &c.n_head);
    req_int(root, "n_embd", path, true, &c.n_embd);
    req_int(root, "vocab_size", path, true, &c.vocab_size);
    if (root.find("n_ctx")) req_int(root, "n_ctx", path, true, &c.n_ctx);
    else if (root.find("n_positions")) req_int(root, "n_positions", path, true, &c.n_ctx);
    else throw std::runtime_error(path + ": missing key 'n_ctx' (or 'n_positions')");
    if (c.n_embd % c.n_head != 0)
        throw std::runtime_error(path + ": n_embd must be divisible by n_head");
    return c;
}

}  // namespace hipgpt2
