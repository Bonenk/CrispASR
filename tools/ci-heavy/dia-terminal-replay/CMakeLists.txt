add_executable(dia-terminal-replay ../dia-cache-intervention/runner.cpp)
target_link_libraries(dia-terminal-replay PRIVATE dia-tts)
add_executable(dia-terminal-guard ../dia-suffix-replay/guard.cpp)
target_link_libraries(dia-terminal-guard PRIVATE ggml ggml-base crispasr_link_ggml_cpu)
