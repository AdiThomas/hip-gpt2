// Minimal JSON reader shared by safetensors.cpp, gpt2_config.cpp and tokenizer.cpp.
// Supports objects, arrays, strings (with \uXXXX incl. surrogate pairs, output as UTF-8),
// numbers (integers kept exactly as int64, others as double), true/false/null.
#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipgpt2 {
namespace json {

struct Value {
    enum Type { Null, Bool, Int, Num, Str, Arr, Obj } type = Null;
    bool b = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<Value> arr;
    // Object members in file order (names may in principle repeat; last wins in find()).
    std::vector<std::pair<std::string, Value>> obj;

    const Value* find(const std::string& key) const {
        if (type != Obj) return nullptr;
        for (size_t k = obj.size(); k-- > 0;)
            if (obj[k].first == key) return &obj[k].second;
        return nullptr;
    }
    bool is_int() const { return type == Int; }
    bool is_number() const { return type == Int || type == Num; }
};

class Parser {
public:
    Parser(const char* p, size_t n) : p_(p), end_(p + n), begin_(p) {}

    Value parse_document() {
        Value v = parse_value(0);
        skip_ws();
        if (p_ != end_) fail("trailing characters after JSON value");
        return v;
    }

private:
    const char* p_;
    const char* end_;
    const char* begin_;

    [[noreturn]] void fail(const std::string& m) const {
        throw std::runtime_error("JSON parse error at offset " + std::to_string(p_ - begin_) + ": " + m);
    }
    void skip_ws() {
        while (p_ < end_ && (*p_ == ' ' || *p_ == '\t' || *p_ == '\n' || *p_ == '\r')) ++p_;
    }
    bool lit(const char* w) {
        const char* q = p_;
        while (*w) {
            if (q >= end_ || *q != *w) return false;
            ++q; ++w;
        }
        p_ = q;
        return true;
    }
    static void put_utf8(std::string& out, uint32_t cp) {
        if (cp < 0x80) out += (char)cp;
        else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
        else if (cp < 0x10000) {
            out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        } else {
            out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
            out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F));
        }
    }
    uint32_t hex4() {
        if (end_ - p_ < 4) fail("truncated \\u escape");
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (uint32_t)(c - '0');
            else if (c >= 'a' && c <= 'f') v |= (uint32_t)(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (uint32_t)(c - 'A' + 10);
            else fail("bad hex digit in \\u escape");
        }
        return v;
    }
    std::string parse_string() {
        if (p_ >= end_ || *p_ != '"') fail("expected string");
        ++p_;
        std::string out;
        for (;;) {
            if (p_ >= end_) fail("unterminated string");
            char c = *p_++;
            if (c == '"') break;
            if ((unsigned char)c < 0x20) fail("control character in string");
            if (c != '\\') { out += c; continue; }
            if (p_ >= end_) fail("unterminated escape");
            char e = *p_++;
            switch (e) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF) {
                        if (end_ - p_ >= 2 && p_[0] == '\\' && p_[1] == 'u') {
                            p_ += 2;
                            uint32_t lo = hex4();
                            if (lo < 0xDC00 || lo > 0xDFFF) fail("bad low surrogate");
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        } else fail("lone high surrogate");
                    } else if (cp >= 0xDC00 && cp <= 0xDFFF) fail("lone low surrogate");
                    put_utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        return out;
    }
    Value parse_number() {
        const char* s = p_;
        bool is_float = false;
        if (p_ < end_ && *p_ == '-') ++p_;
        if (p_ >= end_ || !(*p_ >= '0' && *p_ <= '9')) fail("bad number");
        while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        if (p_ < end_ && *p_ == '.') { is_float = true; ++p_; while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_; }
        if (p_ < end_ && (*p_ == 'e' || *p_ == 'E')) {
            is_float = true; ++p_;
            if (p_ < end_ && (*p_ == '+' || *p_ == '-')) ++p_;
            while (p_ < end_ && *p_ >= '0' && *p_ <= '9') ++p_;
        }
        std::string tok(s, p_);
        Value v;
        if (!is_float) {
            errno = 0;
            char* e = nullptr;
            long long x = std::strtoll(tok.c_str(), &e, 10);
            if (errno == 0) { v.type = Value::Int; v.i = x; v.d = (double)x; return v; }
        }
        v.type = Value::Num;
        v.d = std::strtod(tok.c_str(), nullptr);
        return v;
    }
    Value parse_value(int depth) {
        if (depth > 64) fail("nesting too deep");
        skip_ws();
        if (p_ >= end_) fail("unexpected end of input");
        Value v;
        char c = *p_;
        if (c == '{') {
            ++p_;
            v.type = Value::Obj;
            skip_ws();
            if (p_ < end_ && *p_ == '}') { ++p_; return v; }
            for (;;) {
                skip_ws();
                std::string k = parse_string();
                skip_ws();
                if (p_ >= end_ || *p_ != ':') fail("expected ':'");
                ++p_;
                Value m = parse_value(depth + 1);
                v.obj.emplace_back(std::move(k), std::move(m));
                skip_ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == '}') { ++p_; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            ++p_;
            v.type = Value::Arr;
            skip_ws();
            if (p_ < end_ && *p_ == ']') { ++p_; return v; }
            for (;;) {
                v.arr.push_back(parse_value(depth + 1));
                skip_ws();
                if (p_ < end_ && *p_ == ',') { ++p_; continue; }
                if (p_ < end_ && *p_ == ']') { ++p_; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::Str;
            v.s = parse_string();
        } else if (c == 't') {
            if (!lit("true")) fail("bad literal");
            v.type = Value::Bool; v.b = true;
        } else if (c == 'f') {
            if (!lit("false")) fail("bad literal");
            v.type = Value::Bool; v.b = false;
        } else if (c == 'n') {
            if (!lit("null")) fail("bad literal");
        } else if (c == '-' || (c >= '0' && c <= '9')) {
            v = parse_number();
        } else fail("unexpected character");
        return v;
    }
};

inline Value parse(const char* data, size_t n) { return Parser(data, n).parse_document(); }
inline Value parse(const std::string& s) { return parse(s.data(), s.size()); }

}  // namespace json
}  // namespace hipgpt2
