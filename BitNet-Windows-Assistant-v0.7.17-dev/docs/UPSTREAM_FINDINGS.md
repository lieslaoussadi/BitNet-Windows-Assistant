# Initial upstream findings (2026-10-17)

This is a starting note, not a claim that every finding applies to every machine. Re-check upstream before each release because the repository can change.

- Official repository: https://github.com/microsoft/BitNet
- README documents Python >= 3.10, CMake >= 3.22, Clang >= 18, and on Windows Visual Studio 2022 components for C++, CMake, Git, Clang, and MSBuild/LLVM. It recommends a Visual Studio Developer Command Prompt/PowerShell for Windows builds.
- The official setup script configures CMake, requests `-T ClangCL` for Windows, and currently supplies `-DCMAKE_C_COMPILER=clang` and `-DCMAKE_CXX_COMPILER=clang++` in the shown upstream code. This is a key area to test for toolchain consistency, not a reason to patch blindly.
- A public issue opened in April 2026 reports CMake being unable to locate `clang`/`clang++` on Windows. Another issue describes a Windows compiler-argument conflict and a separate C++ const-correctness compile failure. These are reports from specific configurations and must be reproduced before we adopt any workaround.
- The top-level CMake file adds the `3rdparty/llama.cpp` subdirectory; the official repository must be cloned with submodules as documented.

References:
- https://github.com/microsoft/BitNet/blob/main/README.md
- https://github.com/microsoft/BitNet/blob/main/setup_env.py
- https://github.com/microsoft/BitNet/blob/main/CMakeLists.txt
- https://github.com/microsoft/BitNet/issues/554
- https://github.com/microsoft/BitNet/issues/493
