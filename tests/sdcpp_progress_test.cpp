// Standalone callback regression test; no CUDA execution or sd.cpp linkage.
#define main unused_bench_main
#include "../engines/sdcpp/bench.cpp"
#undef main
#include <cassert>
int main() {
    State st;
    RunTimes r;
    st.cur = &r;
    r.expected_steps = 4;
    on_progress(0, 397, 0, &st); // Before conditioning: ignored.
    assert(r.progress_calls == 0);
    r.cond = 1;
    on_progress(0, 4, 0, &st);
    const double start = r.sampling_start;
    for (int i = 0; i < 37; ++i) on_progress(i * 10, 397, 0, &st);
    assert(r.open_range == "denoise_step_0");
    assert(r.sampling_start == start);
    assert(r.step_end.empty());
    for (int i = 1; i <= 4; ++i) on_progress(i, 4, 0, &st);
    assert(r.progress_calls == 5);
    assert(r.ignored_progress_calls == 37);
    assert(r.step_end.size() == 4);
    assert(r.open_range.empty());
    r.sampling_end = 1;
    on_progress(1, 140, 0, &st); // VAE loading: ignored.
    assert(r.progress_calls == 5);
}
