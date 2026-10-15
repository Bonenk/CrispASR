add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../dia-cache-intervention" "key-base")
add_executable(dia-query-key-graph-guard graph_guard.cpp)
target_link_libraries(dia-query-key-graph-guard PRIVATE ggml ggml-base crispasr_link_ggml_cpu)
