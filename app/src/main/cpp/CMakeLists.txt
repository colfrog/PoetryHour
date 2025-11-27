cmake_minimum_required(VERSION 3.22.1)

project("poetryhour")

find_library(log-lib log)

# 1. Setup Abseil (Dependency)
add_subdirectory(abseil-cpp)

# 2. Setup SentencePiece
# We define the source files manually to avoid building the whole training suite
# and to avoid needing a full Protobuf installation (we use built-in lite mode).

# Use the built-in "Lite" Protobuf (Fixes "missing model_proto.cc")
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)  # Build as Static Lib (easier to link)
set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "" FORCE) # <--- CRITICAL FIX

# Disable bloat (Command line tools, tests, etc.)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(SPM_COVERAGE OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "" FORCE) # Don't try to compile regex rules

add_subdirectory(sentencepiece)
target_link_libraries(sentencepiece-static PUBLIC ${log-lib})

add_library(sentencepiece_native SHARED native-lib.cpp)

# Compiler Flags
target_compile_options(sentencepiece_native PRIVATE -std=c++17 -fexceptions)
target_link_options(sentencepiece_native PRIVATE "-Wl,-z,max-page-size=16384")

# Link Dependencies
target_link_libraries(
        sentencepiece_native
        sentencepiece-static
        log # Android Logging
        android
)
