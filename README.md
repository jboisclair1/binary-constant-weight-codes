# Binary constant-weight codes

## Submission checkpoint · 2026-11-20

101 binary certificates improve lower bounds in the project's 2026-11-18 snapshot of A. E. Brouwer's table [\[1\]](https://aeb.win.tue.nl/codes/Andw.html). Nine attain the matching upper bound. Gain is the number of additional codewords. Each certificate contains one length-$n$ binary word per line, with weight $w$ and pairwise Hamming distance at least $d$.

[ATS 1, ATS 4.8 and exact-link build and run instructions](examples/README.md) · [ATS 4.9 C++ source](src/ats49_pressure.cpp)

Kiermaier, Krčadinac, and Wassermann’s published Steiner design gives $A(42,8,6)=574$ and served as our starting code for several entries below, including the 492-word length-41 closeout [\[2\]](https://doi.org/10.1007/s10623-026-01888-w).

## Exact closeouts

| Certificate | Previous lower bound | Size | Gain | Notes |
|---|---:|---:|---:|---|
| [A(41,8,6)](checkpoints/2026-11-20/a41.8.6.492.txt) | 384 | 492 | +108 | Shorten the published Steiner design on 1 zero coordinate [\[2\]](https://doi.org/10.1007/s10623-026-01888-w). |
| [A(49,8,5)](checkpoints/2026-11-20/a49.8.5.116.txt) | 108 | 116 | +8 | Cyclic development and hole filling. Parent of the length-48 closeout. |
| [A(35,18,16)](checkpoints/2026-11-20/a35.18.16.28.txt) | 22 | 28 | +6 | Group-orbit incidence construction · Previous bound (Christopher D. Rosin) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(69,14,8)](checkpoints/2026-11-20/a69.14.8.77.txt) | 72 | 77 | +5 | Delete a tangent line from the Baer parent [\[3\]](https://doi.org/10.4153/CJM-1987-051-1). For prime powers $r$ and $q=r^2$, $A(q^2-q-r,2q-4,q-1)=q^2-r-1$. |
| [A(44,10,6)](checkpoints/2026-11-20/a44.10.6.58.txt) | 56 | 58 | +2 | Projective-plane restriction · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(45,10,6)](checkpoints/2026-11-20/a45.10.6.60.txt) | 58 | 60 | +2 | Projective-plane restriction · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(48,8,5)](checkpoints/2026-11-20/a48.8.5.105.txt) | 103 | 105 | +2 | Shorten the length-49 code on zero at point 42, removing 11 blocks · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(51,8,5)](checkpoints/2026-11-20/a51.8.5.122.txt) | 121 | 122 | +1 | Nonabelian orbit construction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(52,8,5)](checkpoints/2026-11-20/a52.8.5.124.txt) | 123 | 124 | +1 | Cyclic orbit construction |

## Other lower-bound improvements

| Certificate | Previous lower bound | Size | Gain | Notes |
|---|---:|---:|---:|---|
| [A(56,8,7)](checkpoints/2026-11-20/a56.8.7.5899.txt) | 4,685 | 5,899 | +1,214 | Exact-link reconstruction · Previous bound (Pliny the Liberator) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(57,8,7)](checkpoints/2026-11-20/a57.8.7.6023.txt) | 5,027 | 6,023 | +996 | Exact-link reconstruction |
| [A(44,8,8)](checkpoints/2026-11-20/a44.8.8.7515.txt) | 6,622 | 7,515 | +893 | Exact-link reconstruction |
| [A(43,8,8)](checkpoints/2026-11-20/a43.8.8.6248.txt) | 5,418 | 6,248 | +830 | Exact-link reconstruction |
| [A(55,8,7)](checkpoints/2026-11-20/a55.8.7.5224.txt) | 4,426 | 5,224 | +798 | Exact-link reconstruction · Previous bound (Pliny the Liberator) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(45,8,8)](checkpoints/2026-11-20/a45.8.8.8137.txt) | 7,391 | 8,137 | +746 | Exact-link reconstruction |
| [A(58,8,7)](checkpoints/2026-11-20/a58.8.7.6206.txt) | 5,613 | 6,206 | +593 | Exact-link reconstruction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(54,8,7)](checkpoints/2026-11-20/a54.8.7.4728.txt) | 4,193 | 4,728 | +535 | Exact-link reconstruction · Previous bound (Pliny the Liberator) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(46,8,8)](checkpoints/2026-11-20/a46.8.8.9136.txt) | 8,747 | 9,136 | +389 | Neighbor-code derivation |
| [A(27,4,6)](checkpoints/2026-11-20/a27.4.6.12540.txt) | 12,285 | 12,540 | +255 | ATS 4.8 · Previous bound (Arhan Canli) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(53,8,7)](checkpoints/2026-11-20/a53.8.7.4207.txt) | 3,987 | 4,207 | +220 | Exact-link reconstruction · Previous bound (Pliny the Liberator) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(47,8,8)](checkpoints/2026-11-20/a47.8.8.10672.txt) | 10,535 | 10,672 | +137 | ATS 4.8 |
| [A(43,8,6)](checkpoints/2026-11-20/a43.8.6.574.txt) | 457 | 574 | +117 | Zero-padding of the Steiner design [\[2\]](https://doi.org/10.1007/s10623-026-01888-w) · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(44,8,6)](checkpoints/2026-11-20/a44.8.6.577.txt) | 475 | 577 | +102 | Coordinate extension from the Steiner design [\[2\]](https://doi.org/10.1007/s10623-026-01888-w) · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(64,12,8)](checkpoints/2026-11-20/a64.12.8.616.txt) | 521 | 616 | +95 | Affine cosets of the 77-plane code [\[4\]](https://arxiv.org/abs/1311.0464) [\[5\]](https://arxiv.org/abs/1004.1503) · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(45,8,6)](checkpoints/2026-11-20/a45.8.6.585.txt) | 493 | 585 | +92 | Neutral exact-link exchanges · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(40,8,6)](checkpoints/2026-11-20/a40.8.6.420.txt) | 338 | 420 | +82 | Shorten the published Steiner design on 2 zero coordinates [\[2\]](https://doi.org/10.1007/s10623-026-01888-w) · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(46,8,6)](checkpoints/2026-11-20/a46.8.6.602.txt) | 527 | 602 | +75 | Exact-link exchanges, ATS and conflict-pair splitting · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(59,8,7)](checkpoints/2026-11-20/a59.8.7.6438.txt) | 6,366 | 6,438 | +72 | Exact-link reconstruction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(42,8,8)](checkpoints/2026-11-20/a42.8.8.5204.txt) | 5,136 | 5,204 | +68 | Exact-link reconstruction |
| [A(51,8,6)](checkpoints/2026-11-20/a51.8.6.791.txt) | 728 | 791 | +63 | Continuation · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(50,8,6)](checkpoints/2026-11-20/a50.8.6.763.txt) | 705 | 763 | +58 | Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(49,8,6)](checkpoints/2026-11-20/a49.8.6.740.txt) | 687 | 740 | +53 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(48,8,6)](checkpoints/2026-11-20/a48.8.6.722.txt) | 672 | 722 | +50 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(36,4,5)](checkpoints/2026-11-20/a36.4.5.11247.txt) | 11,207 | 11,247 | +40 | ATS 4.8 · Previous bound (Peter Adel) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(47,8,6)](checkpoints/2026-11-20/a47.8.6.636.txt) | 600 | 636 | +36 | ATS 4.9 · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(39,8,6)](checkpoints/2026-11-20/a39.8.6.357.txt) | 323 | 357 | +34 | Shorten the published Steiner design on 3 zero coordinates [\[2\]](https://doi.org/10.1007/s10623-026-01888-w) · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(29,6,6)](checkpoints/2026-11-20/a29.6.6.1199.txt) | 1,170 | 1,199 | +29 | ATS 4.9 |
| [A(52,8,6)](checkpoints/2026-11-20/a52.8.6.828.txt) | 799 | 828 | +29 |  |
| [A(63,12,8)](checkpoints/2026-11-20/a63.12.8.539.txt) | 520 | 539 | +19 | Affine cosets of the 77-plane code [\[4\]](https://arxiv.org/abs/1311.0464) [\[5\]](https://arxiv.org/abs/1004.1503). Shortened on zero. |
| [A(64,10,8)](checkpoints/2026-11-20/a64.10.8.4798.txt) | 4,779 | 4,798 | +19 | Exact-link reconstruction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(38,14,14)](checkpoints/2026-11-20/a38.14.14.356.txt) | 343 | 356 | +13 | ATS 4.8 · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(62,10,8)](checkpoints/2026-11-20/a62.10.8.3741.txt) | 3,730 | 3,741 | +11 | Exact-link reconstruction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(50,10,8)](checkpoints/2026-11-20/a50.10.8.1201.txt) | 1,193 | 1,201 | +8 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(59,10,7)](checkpoints/2026-11-20/a59.10.7.631.txt) | 623 | 631 | +8 | Reed–Solomon construction with affine-plane blocks · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(53,10,7)](checkpoints/2026-11-20/a53.10.7.441.txt) | 434 | 441 | +7 | Möbius-plane restriction and block packing · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(26,4,5)](checkpoints/2026-11-20/a26.4.5.2822.txt) | 2,816 | 2,822 | +6 | ATS 4.6 |
| [A(48,8,8)](checkpoints/2026-11-20/a48.8.8.12647.txt) | 12,642 | 12,647 | +5 | ATS 4.8 · Previous bound (János Tapolcai) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(52,10,8)](checkpoints/2026-11-20/a52.10.8.1579.txt) | 1,574 | 1,579 | +5 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(54,10,8)](checkpoints/2026-11-20/a54.10.8.2100.txt) | 2,095 | 2,100 | +5 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(58,10,7)](checkpoints/2026-11-20/a58.10.7.617.txt) | 612 | 617 | +5 | Reed–Solomon construction with affine-plane blocks · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(70,14,8)](checkpoints/2026-11-20/a70.14.8.77.txt) | 72 | 77 | +5 | Projective-plane restriction |
| [A(29,8,8)](checkpoints/2026-11-20/a29.8.8.1068.txt) | 1,064 | 1,068 | +4 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(31,8,7)](checkpoints/2026-11-20/a31.8.7.480.txt) | 476 | 480 | +4 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(33,8,7)](checkpoints/2026-11-20/a33.8.7.562.txt) | 558 | 562 | +4 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(51,10,8)](checkpoints/2026-11-20/a51.10.8.1376.txt) | 1,372 | 1,376 | +4 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(52,10,6)](checkpoints/2026-11-20/a52.10.6.82.txt) | 78 | 82 | +4 | Alternating-group orbit search |
| [A(63,10,7)](checkpoints/2026-11-20/a63.10.7.841.txt) | 837 | 841 | +4 | Cyclic backbone construction · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(33,10,7)](checkpoints/2026-11-20/a33.10.7.75.txt) | 72 | 75 | +3 | Möbius-plane restriction |
| [A(42,6,6)](checkpoints/2026-11-20/a42.6.6.5125.txt) | 5,122 | 5,125 | +3 | ATS 4.8 · Previous bound (Arhan Canli) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(51,10,6)](checkpoints/2026-11-20/a51.10.6.79.txt) | 76 | 79 | +3 | Baer-subplane restriction |
| [A(53,10,6)](checkpoints/2026-11-20/a53.10.6.83.txt) | 80 | 83 | +3 | Zero-padding and pair repacking |
| [A(57,10,7)](checkpoints/2026-11-20/a57.10.7.603.txt) | 600 | 603 | +3 | Reed–Solomon construction with affine-plane blocks · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(59,8,5)](checkpoints/2026-11-20/a59.8.5.161.txt) | 158 | 161 | +3 | ATS 4.8 |
| [A(61,12,7)](checkpoints/2026-11-20/a61.12.7.80.txt) | 77 | 80 | +3 | Projective-plane restriction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(30,8,8)](checkpoints/2026-11-20/a30.8.8.1168.txt) | 1,166 | 1,168 | +2 | ATS 4.9 · Previous bound (Pliny the Liberator) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(34,8,7)](checkpoints/2026-11-20/a34.8.7.606.txt) | 604 | 606 | +2 | ATS 4.9 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(36,10,7)](checkpoints/2026-11-20/a36.10.7.118.txt) | 116 | 118 | +2 | Möbius-plane restriction · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(51,10,7)](checkpoints/2026-11-20/a51.10.7.413.txt) | 411 | 413 | +2 | Möbius-plane restriction · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(52,10,7)](checkpoints/2026-11-20/a52.10.7.423.txt) | 421 | 423 | +2 | Möbius-plane restriction and block packing · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(54,12,8)](checkpoints/2026-11-20/a54.12.8.352.txt) | 350 | 352 | +2 | ATS 4.8 |
| [A(55,10,7)](checkpoints/2026-11-20/a55.10.7.521.txt) | 519 | 521 | +2 | ATS 4.8 · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(55,10,8)](checkpoints/2026-11-20/a55.10.8.2424.txt) | 2,422 | 2,424 | +2 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(56,8,6)](checkpoints/2026-11-20/a56.8.6.1166.txt) | 1,164 | 1,166 | +2 | ATS 4.9 · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(56,10,7)](checkpoints/2026-11-20/a56.10.7.589.txt) | 587 | 589 | +2 | Cyclic backbone construction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(58,10,6)](checkpoints/2026-11-20/a58.10.6.99.txt) | 97 | 99 | +2 | Group-orbit block packing · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(58,12,7)](checkpoints/2026-11-20/a58.12.7.73.txt) | 71 | 73 | +2 | Projective-plane restriction |
| [A(60,12,7)](checkpoints/2026-11-20/a60.12.7.76.txt) | 74 | 76 | +2 | Baer-subplane restriction · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(63,10,8)](checkpoints/2026-11-20/a63.10.8.4230.txt) | 4,228 | 4,230 | +2 | Exact-link reconstruction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(29,10,13)](checkpoints/2026-11-20/a29.10.13.760.txt) | 759 | 760 | +1 | Coordinate extension · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(30,8,7)](checkpoints/2026-11-20/a30.8.7.406.txt) | 405 | 406 | +1 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(30,10,13)](checkpoints/2026-11-20/a30.10.13.941.txt) | 940 | 941 | +1 | ATS 4.9 · Previous bound (Antoine Olivier) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(34,10,7)](checkpoints/2026-11-20/a34.10.7.86.txt) | 85 | 86 | +1 | Möbius-plane restriction · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(35,14,14)](checkpoints/2026-11-20/a35.14.14.164.txt) | 163 | 164 | +1 | ATS 4.8 |
| [A(36,6,5)](checkpoints/2026-11-20/a36.6.5.626.txt) | 625 | 626 | +1 | ATS 4.8 · Previous bound (Arhan Canli) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(39,8,5)](checkpoints/2026-11-20/a39.8.5.69.txt) | 68 | 69 | +1 | Cyclic orbit packing · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(41,8,7)](checkpoints/2026-11-20/a41.8.7.1296.txt) | 1,295 | 1,296 | +1 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(43,10,8)](checkpoints/2026-11-20/a43.10.8.471.txt) | 470 | 471 | +1 | ATS 4.8 · Previous bound (William Echols) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(46,10,6)](checkpoints/2026-11-20/a46.10.6.61.txt) | 60 | 61 | +1 | Projective-plane restriction · Previous bound (Ankan Sadhu) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(46,10,7)](checkpoints/2026-11-20/a46.10.7.316.txt) | 315 | 316 | +1 | ATS 4.8 · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(47,6,5)](checkpoints/2026-11-20/a47.6.5.1392.txt) | 1,391 | 1,392 | +1 | ATS 4.8 · Previous bound (Arhan Canli) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(47,10,7)](checkpoints/2026-11-20/a47.10.7.334.txt) | 333 | 334 | +1 | ATS 4.9 · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(48,8,7)](checkpoints/2026-11-20/a48.8.7.2873.txt) | 2,872 | 2,873 | +1 | ATS 4.8 |
| [A(53,12,8)](checkpoints/2026-11-20/a53.12.8.351.txt) | 350 | 351 | +1 | Coordinate extension |
| [A(54,12,7)](checkpoints/2026-11-20/a54.12.7.60.txt) | 59 | 60 | +1 | Projective-plane restriction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(57,10,6)](checkpoints/2026-11-20/a57.10.6.96.txt) | 95 | 96 | +1 | Group-orbit block packing |
| [A(59,10,6)](checkpoints/2026-11-20/a59.10.6.100.txt) | 99 | 100 | +1 | Group-orbit block packing · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(59,12,7)](checkpoints/2026-11-20/a59.12.7.73.txt) | 72 | 73 | +1 | Projective-plane restriction · Previous bound (Jordan Boisclair) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(60,10,6)](checkpoints/2026-11-20/a60.10.6.106.txt) | 105 | 106 | +1 | Group-orbit block packing |
| [A(61,10,8)](checkpoints/2026-11-20/a61.10.8.3308.txt) | 3,307 | 3,308 | +1 | ATS 4.8 · Previous bound (Guan-Ying Huang) [\[1\]](https://aeb.win.tue.nl/codes/Andw.html) |
| [A(62,10,6)](checkpoints/2026-11-20/a62.10.6.115.txt) | 114 | 115 | +1 | Group-orbit block packing |
| [A(62,12,7)](checkpoints/2026-11-20/a62.12.7.80.txt) | 79 | 80 | +1 | Projective-plane restriction |


## Earlier checkpoints

- [August 30, 2026 · 90 codes](checkpoints/2026-08-30/README.md)
- [August 22, 2026 · 18 codes](checkpoints/2026-08-22/README.md)

## References

1. A. E. Brouwer, [Bounds for binary constant weight codes](https://aeb.win.tue.nl/codes/Andw.html).
2. [Steiner 3-designs as extensions](https://doi.org/10.1007/s10623-026-01888-w). Source of the Steiner system on 42 points.
3. D. Jungnickel, [Divisible Semiplanes, Arcs, and Relative Difference Sets](https://doi.org/10.4153/CJM-1987-051-1). Background for the Baer parent geometry.
4. T. Honold, M. Kiermaier and S. Kurz, [Optimal Binary Subspace Codes of Length 6, Constant Dimension 3 and Minimum Distance 4](https://arxiv.org/abs/1311.0464).
5. T. Etzion and A. Vardy, [A New Construction for Constant Weight Codes](https://arxiv.org/abs/1004.1503).
