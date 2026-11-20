# Running the example searches

Run these commands from the repository root in an **x64 Native Tools Command Prompt for Visual Studio 2022**. The input seeds use one-based coordinate supports.

## ATS 1 (Augmenting Trade Search)

This archived standalone C# extraction contains the original shared-reservoir search and its v2 extension. The command below selects v1 by using a single reservoir. It requires the .NET 9 SDK and restores offline.

~~~bat
dotnet restore src/ats1/Ats.csproj --configfile src/ats1/NuGet.Config
dotnet build src/ats1/Ats.csproj -c Release --no-restore
dotnet src/ats1/bin/Release/net9.0/ats.dll search --input examples/seeds/a27.6.5.261.txt --n 27 --d 6 --w 5 --reservoir 27 --max-results 9 --solutions-per-candidate 9 --output result.txt
~~~

From the 261-word seed, the first output (result.1.txt) reproduces the August 22 certificate’s 262 codewords, with minimum distance six. [Seed](seeds/a27.6.5.261.txt) · [August certificate](../checkpoints/2026-08-22/a27.6.5.262.txt) · [Solver source](../src/ats1/Program.cs)

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
