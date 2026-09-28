// Benchmark harness for stable-diffusion.cpp: load the model once, then generate N images.
//
// This mirrors benchmark() in scripts/run_flux.py for a C++ engine. It is driven by
// scripts/run_sdcpp.py, which passes every setting explicitly and turns the output into the
// standard run-directory files. sd.cpp's own source is not modified; stage boundaries come from
// its public log and progress callbacks:
//
//   generate start ─ text_encode ─ "get_learned_condition completed"
//                  ─ (other: noise init) ─ progress(step=0)
//                  ─ denoise_step_i ─ progress(step=i+1)            (i = 0..steps-1)
//                  ─ "sampling completed" ─ vae_decode ─ "decode_first_stage completed"
//                  ─ (other: conversion to uint8, return) ─ generate end
//
// ggml computes each graph synchronously, so a host timestamp taken when a callback fires is
// also the point at which that stage's GPU work has finished. Timestamps are steady_clock
// (CLOCK_MONOTONIC on Linux), the same clock as Python's time.perf_counter(), so the wrapper can
// align them with its NVML memory samples.
//
// With -DSD_BENCH_NVTX, the same boundaries are emitted as NVTX ranges for Nsight Systems.
//
// Outputs (in --out-dir):
//   results.jsonl   one JSON object for the load, then one per generation
//   sdcpp.log       every sd.cpp log line, prefixed with its timestamp (s)
//   raw/run-<i>.rgb raw RGB bytes of every generated image (hashed and converted by the wrapper)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#ifdef SD_BENCH_NVTX
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) ((void)0)
#define NVTX_POP() ((void)0)
#endif

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct RunTimes {
    double start = 0, cond = 0, sampling_start = 0, sampling_end = 0, decode_end = 0, end = 0;
    std::vector<double> step_end;
    int progress_calls = 0;
    int cond_cache_hits = 0;
    std::string open_range;  // name of the NVTX stage range currently open
};

struct State {
    FILE* log = nullptr;
    RunTimes* cur = nullptr;  // non-null while a generation is running
};

static void open_stage(RunTimes* r, const std::string& name) {
    if (!r->open_range.empty()) NVTX_POP();
    r->open_range = name;
    if (!name.empty()) NVTX_PUSH(name.c_str());
}

static void on_log(enum sd_log_level_t level, const char* text, void* data) {
    auto* st = static_cast<State*>(data);
    double t = now_s();
    if (st->log) {
        std::fprintf(st->log, "%.6f [%d] %s", t, (int)level, text);
        size_t n = std::strlen(text);
        if (n == 0 || text[n - 1] != '\n') std::fputc('\n', st->log);
    }
    RunTimes* r = st->cur;
    if (!r) return;
    if (std::strstr(text, "get_learned_condition completed")) {
        r->cond = t;
        open_stage(r, "");
    } else if (std::strstr(text, "sampling completed") && !std::strstr(text, "hires")) {
        r->sampling_end = t;
        open_stage(r, "vae_decode");
    } else if (std::strstr(text, "decode_first_stage completed")) {
        r->decode_end = t;
        open_stage(r, "");
    } else if (std::strstr(text, "conditioning cache hit") || std::strstr(text, "condition cache hit")) {
        r->cond_cache_hits++;
    }
}

static void on_progress(int step, int steps, float, void* data) {
    auto* st = static_cast<State*>(data);
    RunTimes* r = st->cur;
    if (!r || r->cond == 0 || r->sampling_end != 0) return;  // only count calls inside sampling
    double t = now_s();
    r->progress_calls++;
    if (step == 0) {
        r->sampling_start = t;
    } else {
        r->step_end.push_back(t);
    }
    if (step < steps) {
        open_stage(r, "denoise_step_" + std::to_string(step));
    } else {
        open_stage(r, "");
    }
}

static std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        if (k.rfind("--", 0) != 0 || i + 1 >= argc) {
            std::fprintf(stderr, "bad argument %s (expected --key value)\n", k.c_str());
            std::exit(2);
        }
        a[k.substr(2)] = argv[++i];
    }
    return a;
}

static std::string need(std::map<std::string, std::string>& a, const char* k) {
    auto it = a.find(k);
    if (it == a.end()) {
        std::fprintf(stderr, "missing required --%s\n", k);
        std::exit(2);
    }
    return it->second;
}

static bool flag(std::map<std::string, std::string>& a, const char* k) {
    return need(a, k) == "1";
}

int main(int argc, char** argv) {
    auto a = parse_args(argc, argv);
    const std::string out_dir = need(a, "out-dir");
    const int runs = std::stoi(need(a, "runs"));
    const int steps = std::stoi(need(a, "steps"));

    State st;
    st.log = std::fopen((out_dir + "/sdcpp.log").c_str(), "w");
    std::ofstream res(out_dir + "/results.jsonl");
    std::system(("mkdir -p '" + out_dir + "/raw'").c_str());
    sd_set_log_callback(on_log, &st);
    sd_set_progress_callback(on_progress, &st);

    // Every field that affects execution is set explicitly from the wrapper's arguments.
    const std::string diffusion = need(a, "diffusion-model"), llm = need(a, "llm"), vae = need(a, "vae");
    const std::string backend = need(a, "backend"), params_backend = need(a, "params-backend");
    sd_ctx_params_t cp;
    sd_ctx_params_init(&cp);
    cp.diffusion_model_path = diffusion.c_str();
    cp.llm_path = llm.c_str();
    cp.vae_path = vae.c_str();
    cp.n_threads = std::stoi(need(a, "threads"));
    cp.wtype = SD_TYPE_COUNT;  // keep the weight file's type (no conversion)
    cp.enable_mmap = flag(a, "mmap");
    cp.flash_attn = flag(a, "fa");
    cp.diffusion_flash_attn = flag(a, "diffusion-fa");
    cp.backend = backend.c_str();
    cp.params_backend = params_backend.c_str();
    cp.auto_fit = flag(a, "auto-fit");
    cp.eager_load = flag(a, "eager-load");
    cp.disable_segmented_compute = flag(a, "disable-segmented-compute");
    cp.disable_prefetch = flag(a, "disable-prefetch");
    cp.conditioning_cache_size = std::stoi(need(a, "conditioning-cache-size"));

    const char* ctx_str = sd_ctx_params_to_str(&cp);
    if (st.log) std::fprintf(st.log, "%.6f [ctx_params]\n%s\n", now_s(), ctx_str);

    NVTX_PUSH("load");
    double t0 = now_s();
    sd_ctx_t* ctx = new_sd_ctx(&cp);
    double t1 = now_s();
    NVTX_POP();
    if (!ctx) {
        res << "{\"event\":\"load\",\"ok\":false}\n";
        std::fprintf(stderr, "new_sd_ctx failed; see sdcpp.log\n");
        return 1;
    }
    res << "{\"event\":\"load\",\"ok\":true,\"t_start\":" << std::to_string(t0) << ",\"t_end\":" << std::to_string(t1)
        << ",\"model_version\":\"" << sd_get_model_version_name(ctx) << "\"}\n";
    res.flush();

    const std::string prompt = need(a, "prompt");
    sd_img_gen_params_t gp;
    sd_img_gen_params_init(&gp);
    gp.prompt = prompt.c_str();
    gp.negative_prompt = "";
    gp.width = std::stoi(need(a, "width"));
    gp.height = std::stoi(need(a, "height"));
    gp.seed = std::stoll(need(a, "seed"));
    gp.batch_count = 1;
    gp.sample_params.sample_steps = steps;
    gp.sample_params.guidance.txt_cfg = std::stof(need(a, "cfg-scale"));
    if (a.count("sampling-method")) {
        gp.sample_params.sample_method = str_to_sample_method(a.at("sampling-method").c_str());
        if (gp.sample_params.sample_method == SAMPLE_METHOD_COUNT) return 2;
    }
    if (a.count("scheduler")) {
        gp.sample_params.scheduler = str_to_scheduler(a.at("scheduler").c_str());
        if (gp.sample_params.scheduler == SCHEDULER_COUNT) return 2;
    }
    // sample_method / scheduler stay at *_COUNT: the library resolves the model's defaults, as
    // sd-cli does. The resolved values appear in sdcpp.log and are recorded by the wrapper.
    gp.vae_tiling_params.enabled = false;
    gp.cache.mode = SD_CACHE_DISABLED;

    const char* gp_str = sd_img_gen_params_to_str(&gp);
    if (st.log) std::fprintf(st.log, "%.6f [img_gen_params]\n%s\n", now_s(), gp_str);

    for (int i = 0; i < runs; i++) {
        RunTimes r;
        st.cur = &r;
        sd_image_t* images = nullptr;
        int n_images = 0;
        NVTX_PUSH("generate");
        r.start = now_s();
        open_stage(&r, "text_encode");
        bool ok = generate_image(ctx, &gp, &images, &n_images);
        r.end = now_s();
        open_stage(&r, "");
        NVTX_POP();
        st.cur = nullptr;

        res << "{\"event\":\"generate\",\"run_index\":" << i << ",\"ok\":" << (ok && n_images == 1 ? "true" : "false")
            << ",\"t_start\":" << std::to_string(r.start) << ",\"t_cond\":" << std::to_string(r.cond)
            << ",\"t_sampling_start\":" << std::to_string(r.sampling_start) << ",\"t_step_end\":[";
        for (size_t k = 0; k < r.step_end.size(); k++) res << (k ? "," : "") << std::to_string(r.step_end[k]);
        res << "],\"t_sampling_end\":" << std::to_string(r.sampling_end) << ",\"t_decode_end\":"
            << std::to_string(r.decode_end) << ",\"t_end\":" << std::to_string(r.end)
            << ",\"progress_calls\":" << r.progress_calls << ",\"cond_cache_hits\":" << r.cond_cache_hits;
        if (ok && n_images == 1) {
            const sd_image_t& im = images[0];
            res << ",\"width\":" << im.width << ",\"height\":" << im.height << ",\"channels\":" << im.channel;
            // Written after timing, so it is not part of the measured generation.
            std::ofstream raw(out_dir + "/raw/run-" + std::to_string(i) + ".rgb", std::ios::binary);
            raw.write(reinterpret_cast<const char*>(im.data), (std::streamsize)im.width * im.height * im.channel);
        }
        res << "}\n";
        res.flush();
        if (images) {
            for (int k = 0; k < n_images; k++) std::free(images[k].data);
            std::free(images);
        }
        if (!ok) break;
    }
    free_sd_ctx(ctx);
    if (st.log) std::fclose(st.log);
    return 0;
}
