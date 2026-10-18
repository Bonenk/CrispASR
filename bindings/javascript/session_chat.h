#pragma once
#include <emscripten/bind.h>

#if !defined(CRISPASR_WASM_CHAT) || defined(__EMSCRIPTEN_PTHREADS__)
static void binding_register_chat() {
    emscripten::function("chatAvailable", emscripten::optional_override([] { return false; }));
}
#else
#include "crispasr_chat.h"
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// JS exceptions are caught before returning through the native decoder frames.
// clang-format off
EM_JS(int, binding_chat_callback, (uintptr_t handle, const char* text), {
    const state = Emval.toValue(handle);
    if (state.failed)
        return 0;
    Module['crispasrInCallback'] = true;
    try {
        if (text) {
            if (typeof state.token === 'function') {
                let end = text;
                while (HEAPU8[end])
                    ++end;
                state.token(HEAPU8.slice(text, end));
            }
            return 1;
        }
        return typeof state.abort !== 'function' || !!state.abort();
    } catch (e) {
        state.failed = true;
        state.error = e;
        return 0;
    } finally {
        Module['crispasrInCallback'] = false;
    }
});
// clang-format on
namespace binding_chat {
using emscripten::val;
static bool present(const val& object, const char* key) {
    return !object.isNull() && !object.isUndefined() && !object[key].isUndefined();
}
static std::string string(const val& value) {
    auto out = value.as<std::string>();
    if (out.find('\0') != std::string::npos)
        throw std::invalid_argument("Embedded NUL in chat string");
    return out;
}
struct OpenOptions {
    crispasr_chat_open_params native{};
    std::string chat_template;
    explicit OpenOptions(const val& options) {
        crispasr_chat_open_params_default(&native);
#define CHAT_OPEN_INT(js, field)                                                                                       \
    if (present(options, js))                                                                                          \
        native.field = options[js].as<int>();
        CHAT_OPEN_INT("threads", n_threads)
        CHAT_OPEN_INT("batchThreads", n_threads_batch)
        CHAT_OPEN_INT("contextSize", n_ctx)
        CHAT_OPEN_INT("batchSize", n_batch)
        CHAT_OPEN_INT("microBatchSize", n_ubatch)
        CHAT_OPEN_INT("gpuLayers", n_gpu_layers)
#undef CHAT_OPEN_INT
        if (present(options, "useMmap"))
            native.use_mmap = options["useMmap"].as<bool>();
        if (present(options, "useMlock"))
            native.use_mlock = options["useMlock"].as<bool>();
        if (present(options, "embeddings"))
            native.embeddings = options["embeddings"].as<bool>();
        if (present(options, "chatTemplate")) {
            chat_template = string(options["chatTemplate"]);
            native.chat_template = chat_template.c_str();
        }
    }
};
struct Messages {
    std::vector<std::string> roles, contents;
    std::vector<crispasr_chat_message> native;
    explicit Messages(const val& input) {
        size_t n = input["length"].as<size_t>();
        roles.reserve(n);
        contents.reserve(n);
        native.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            roles.push_back(string(input[i]["role"]));
            contents.push_back(string(input[i]["content"]));
        }
        for (size_t i = 0; i < n; ++i)
            native.push_back({roles[i].c_str(), contents[i].c_str()});
    }
};
struct GenerateOptions {
    crispasr_chat_generate_params native{};
    std::vector<std::string> stops;
    std::vector<const char*> pointers;
    explicit GenerateOptions(const val& options) {
        crispasr_chat_generate_params_default(&native);
#define CHAT_GENERATE(js, field, type)                                                                                 \
    if (present(options, js))                                                                                          \
        native.field = options[js].as<type>();
        CHAT_GENERATE("maxTokens", max_tokens, int)
        CHAT_GENERATE("temperature", temperature, float)
        CHAT_GENERATE("topK", top_k, int)
        CHAT_GENERATE("topP", top_p, float)
        CHAT_GENERATE("minP", min_p, float)
        CHAT_GENERATE("repeatPenalty", repeat_penalty, float)
        CHAT_GENERATE("repeatLastN", repeat_last_n, int)
        CHAT_GENERATE("seed", seed, uint32_t)
        CHAT_GENERATE("prefillOnly", prefill_only, bool)
#undef CHAT_GENERATE
        if (present(options, "stop")) {
            size_t n = options["stop"]["length"].as<size_t>();
            stops.reserve(n);
            pointers.reserve(n);
            for (size_t i = 0; i < n; ++i)
                stops.push_back(string(options["stop"][i]));
            for (auto& stop : stops)
                pointers.push_back(stop.c_str());
            native.stop = pointers.data();
            native.n_stop = pointers.size();
        }
    }
};
struct State {
    crispasr_chat_session_t native = nullptr;
    val callbacks = val::object();
    State() {
        callbacks.set("abort", val::undefined());
        callbacks.set("token", val::undefined());
        callbacks.set("failed", false);
    }
    ~State() {
        if (native)
            crispasr_chat_close(native);
    }
};
static bool abort(void* user) {
    return binding_chat_callback(reinterpret_cast<uintptr_t>(static_cast<State*>(user)->callbacks.as_handle()),
                                 nullptr) != 0;
}
static void token(const char* text, void* user) {
    binding_chat_callback(reinterpret_cast<uintptr_t>(static_cast<State*>(user)->callbacks.as_handle()), text);
}
static void require_success(int rc, const crispasr_chat_error& err) {
    if (rc != 0)
        throw std::runtime_error(std::string("Chat failed: ") + err.message);
}
class Session {
    std::shared_ptr<State> state;
    std::shared_ptr<State> checked() const {
        binding_check_reentry();
        if (!state)
            throw std::runtime_error("Chat session is closed");
        return state;
    }

public:
    Session(const std::string& path, const val& options) {
        binding_check_reentry();
        if (path.find('\0') != std::string::npos)
            throw std::invalid_argument("Embedded NUL in model path");
        OpenOptions p(options);
        auto next = std::make_shared<State>();
        crispasr_chat_error err{};
        next->native = crispasr_chat_open(path.c_str(), &p.native, &err);
        if (!next->native)
            throw std::runtime_error(std::string("Chat open failed: ") + err.message);
        crispasr_chat_set_abort_callback(next->native, abort, next.get());
        state = std::move(next);
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    void close() {
        binding_check_reentry();
        state.reset();
    }
    void reset() {
        auto s = checked();
        crispasr_chat_error err{};
        require_success(crispasr_chat_reset(s->native, &err), err);
    }
    void set_abort_callback(const val& callback) {
        auto s = checked();
        s->callbacks.set("abort", callback);
    }
    std::string template_name() const {
        auto s = checked();
        const char* p = crispasr_chat_template_name(s->native);
        return p ? p : "";
    }
    int context_size() const {
        auto s = checked();
        return crispasr_chat_n_ctx(s->native);
    }
    int count_tokens(const val& messages) const {
        auto s = checked();
        Messages m(messages);
        crispasr_chat_error err{};
        int n = crispasr_chat_count_tokens(s->native, m.native.data(), m.native.size(), &err);
        if (n < 0)
            require_success(n, err);
        return n;
    }
    val generate(const val& messages, const val& options, const val& callback) {
        // Keep state alive even if a JS token callback deletes its Embind owner.
        auto s = checked();
        Messages m(messages);
        GenerateOptions p(options);
        auto callbacks = s->callbacks;
        callbacks.set("failed", false);
        callbacks.set("error", val::undefined());
        callbacks.set("token", callback);
        crispasr_chat_error err{};
        int rc = 0;
        std::string text;
        if (callback.isNull() || callback.isUndefined()) {
            std::unique_ptr<char, decltype(&crispasr_chat_string_free)> reply(
                crispasr_chat_generate(s->native, m.native.data(), m.native.size(), &p.native, &err),
                crispasr_chat_string_free);
            if (reply)
                text = reply.get();
            else
                rc = err.code ? err.code : -1;
        } else
            rc = crispasr_chat_generate_stream(s->native, m.native.data(), m.native.size(), &p.native, token, s.get(),
                                               &err);
        callbacks.set("token", val::undefined());
        auto out = val::object();
        out.set("status", rc);
        out.set("text", text);
        out.set("message", std::string(err.message));
        out.set("failed", callbacks["failed"]);
        out.set("error", callbacks["error"]);
        return out;
    }
};
} // namespace binding_chat
static void binding_register_chat() {
    using namespace emscripten;
    function("chatAvailable", optional_override([] { return true; }));
    function("chatMemoryEstimate", optional_override([](const std::string& model, const val& options) {
                 binding_check_reentry();
                 binding_chat::OpenOptions p(options);
                 crispasr_chat_error err{};
                 size_t n = crispasr_chat_memory_estimate(model.c_str(), &p.native, &err);
                 if (n == 0 && err.code)
                     binding_chat::require_success(err.code, err);
                 return double(n);
             }));
    function("chatAiDisclosure", optional_override([] {
                 binding_check_reentry();
                 const char* p = crispasr_chat_ai_disclosure_text();
                 return std::string(p ? p : "");
             }));
    class_<binding_chat::Session>("ChatSession")
        .constructor<const std::string&, const val&>()
        .function("close", &binding_chat::Session::close)
        .function("reset", &binding_chat::Session::reset)
        .function("setAbortCallback", &binding_chat::Session::set_abort_callback)
        .function("templateName", &binding_chat::Session::template_name)
        .function("contextSize", &binding_chat::Session::context_size)
        .function("countTokens", &binding_chat::Session::count_tokens)
        .function("generate", &binding_chat::Session::generate);
}
#endif
