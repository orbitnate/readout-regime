# LaTeX source

- **`main.tex`** — the self-contained LaTeX source for the paper.
- **`fig_attractor.pdf`** — Figure 2 (the off-axis attractor plot; vector).
- **`main.pdf`** — the compiled paper (19 pages).

## Compile

The document uses XeLaTeX-style typesetting (`fontspec` + `unicode-math`, Latin Modern).

- **Tectonic** (no TeX install required):
  ```
  tectonic main.tex
  ```
- **Overleaf**: upload `main.tex` and `fig_attractor.pdf`, set the compiler to **XeLaTeX**, and click Compile.
- **Local TeX Live**: `xelatex main.tex` (run twice to resolve references).

## arXiv submission

Upload `main.tex` and `fig_attractor.pdf`. The source compiles with XeLaTeX; if arXiv
does not auto-detect it, select the **XeLaTeX** engine (the document loads `fontspec`,
which requires XeLaTeX or LuaLaTeX rather than the default pdfLaTeX).
