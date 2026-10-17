add_executable(dia-suffix-replay ../dia-cache-intervention/runner.cpp)
target_link_libraries(dia-suffix-replay PRIVATE dia-tts)
add_executable(dia-suffix-guard guard.cpp)
target_link_libraries(dia-suffix-guard PRIVATE ggml ggml-base crispasr_link_ggml_cpu)
