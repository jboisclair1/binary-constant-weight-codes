@echo off
setlocal
pushd "%~dp0" || exit /b 1
set "ats_arch=%~1"
if not defined ats_arch set "ats_arch=89"
cl /nologo /std:c++20 /O2 /EHsc /c ATS_v4_8_branch_lookback.cpp || exit /b 1
nvcc -std=c++17 -O3 -arch=sm_%ats_arch% -Xcompiler=/EHsc,/O2 -c ATS_v4_5_cuda.cu -o cuda_backend.obj || exit /b 1
nvcc -arch=sm_%ats_arch% ATS_v4_8_branch_lookback.obj cuda_backend.obj -o ATS_v4_8_cuda.exe || exit /b 1
copy /y ATS_v4_8_cuda.exe ATS_v4_8_cuda_sm%ats_arch%.exe >nul || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /Ihalo_compat ATS_v4_6_halo_novelty.cpp || exit /b 1
cl /nologo /std:c++20 /O2 /EHsc /Ihalo_compat ATS_v4_6_halo_expand.cpp || exit /b 1
popd
