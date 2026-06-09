cmake_minimum_required(VERSION 3.18)
project(pyai LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Build type default ────────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ── Optional CUDA ─────────────────────────────────────────────────────────────
option(WITH_CUDA "Build with CUDA GPU support" OFF)

if(WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # Generate code for common architectures; add more as needed
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89 90)
  add_compile_definitions(WITH_CUDA)
  message(STATUS "CUDA enabled — architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA disabled — CPU-only build")
endif()

# ── Platform-specific compiler flags ─────────────────────────────────────────
if(MSVC)
  add_compile_options(/O2 /W3 /MP /utf-8)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
else()
  add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)
  if(UNIX AND NOT APPLE)
    add_link_options(-pthread)
  endif()
endif()

# ── Sources ───────────────────────────────────────────────────────────────────
set(SOURCES
  src/main.cpp
  src/vocab.cpp
  src/model.cpp
  src/trainer.cpp
  src/generator.cpp
  src/tokenizer.cpp
  src/settings.cpp
  src/downloader.cpp
  src/utils.cpp
)

if(WITH_CUDA)
  list(APPEND SOURCES cuda/cuda_ops.cu)
endif()

add_executable(pyai ${SOURCES})

target_include_directories(pyai PRIVATE include)

if(WITH_CUDA)
  target_include_directories(pyai PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(pyai PRIVATE CUDA::cublas CUDA::curand)
  set_target_properties(pyai PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# ── zlib for gzip (model save/load) ──────────────────────────────────────────
find_package(ZLIB REQUIRED)
target_link_libraries(pyai PRIVATE ZLIB::ZLIB)

# ── Threading ─────────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
target_link_libraries(pyai PRIVATE Threads::Threads)

# Windows: link ws2_32 for network I/O (downloader)
if(WIN32)
  target_link_libraries(pyai PRIVATE ws2_32 winhttp)
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS pyai DESTINATION bin)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
