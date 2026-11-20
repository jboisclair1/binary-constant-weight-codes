# Running the example searches

Run these commands from the repository root in an **x64 Native Tools Command Prompt for Visual Studio 2022**. Both inputs use one-based coordinate supports.

## ATS 4.8 HALO + ACPS

The example starts from the 625-word $A(36,6,5)$ seed credited to Arhan Canli in the comparison table. The test run produced 626 verified words. [Input seed](seeds/a36.6.5.625.txt) · [Original table seed](https://aeb.win.tue.nl/codes/cwc/d6/a36.6.5.625H) · [Checkpoint certificate](../checkpoints/2026-11-20/a36.6.5.626.txt)

Use Python 3.10 or newer and an NVIDIA CUDA toolkit compatible with the host compiler. The tested build used CUDA 12.9 and an RTX 4070 Ti (compute capability 8.9). The build argument selects the CUDA architecture. The driver recognizes the GPU builds for 8.6 and 8.9.

~~~bat
call src\ats48\build.cmd 89
python src\ats48\ATS_v4_8_halo_acps.py --seed examples\seeds\a36.6.5.625.txt --n 36 --d 6 --w 5 --seconds 30 --threads 12 --output-dir ats-result --rng-seed 4800111 --runtime-policy deadline --max-chain-gain 1
~~~

## Exact-link search

The example starts from the 4,615-word $A(54,8,7)$ input used in the original exact-link campaign. The command below produced 4,696 verified words in 60 seconds. The original campaign reached 4,728. The solver saves its best code to the output file. Final size can vary with runtime and hardware. [Input seed](seeds/a54.8.7.4615.txt) · [Checkpoint certificate](../checkpoints/2026-11-20/a54.8.7.4728.txt)

~~~bat
cl /nologo /std:c++20 /EHsc /O2 src\exact-link\link_reconstruct.cpp /Feexact-link.exe
exact-link.exe examples\seeds\a54.8.7.4615.txt 54 8 7 1 60 link-result.txt 0.2
~~~

The arguments after the seed file are length, distance, weight, random seed, seconds, output file, and probability of accepting a size-preserving replacement.
