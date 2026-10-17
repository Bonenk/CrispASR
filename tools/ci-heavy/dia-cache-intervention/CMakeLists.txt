add_executable(dia-key-intervention runner.cpp)
target_link_libraries(dia-key-intervention PRIVATE dia-tts)
add_executable(dia-key-graph-guard graph_guard.cpp)
target_link_libraries(dia-key-graph-guard PRIVATE ggml ggml-base crispasr_link_ggml_cpu)
