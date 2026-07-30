# JAIR submission version

`precision_patches_jair.tex` — the JAIR-formatted version of the paper.
The **body text is identical** to the arXiv version
(`../precision_patches.tex`); only these differ, per the JAIR Author Kit:

- `\documentclass[manuscript, screen, review]{jair}` (ACM-based `jair.cls`)
- structured abstract (Background / Objectives / Methods / Results / Conclusions)
- JAIR title/author block, `\setcopyright{cc}`
- bibliography as `refs.bib`
- **Appendix A: filled-in JAIR Reproducibility Checklist**

## Build

```bash
tectonic precision_patches_jair.tex      # or: xelatex + bibtex + xelatex x2
```

Two environment notes:

1. **Fonts.** `acmart.cls` under XeLaTeX/LuaLaTeX needs the Libertinus OTF
   fonts, requested by lowercase filename (`libertinusmath-regular.otf`).
   Download from <https://github.com/alerque/libertinus/releases>, install to
   `~/.local/share/fonts/`, and add lowercase copies — or compile with
   pdfLaTeX, which uses the Type1 path and avoids this entirely.
2. **Bibliography.** The Author Kit ships a BibLaTeX + `biber` setup. `biber`
   was unavailable in this build environment, so this file uses acmart's
   native natbib + BibTeX path (`\bibliographystyle{ACM-Reference-Format}`),
   which produces the same author-year format. If your environment has
   `biber`, you may restore the kit's original block — see the comment in the
   preamble.

Class files (`jair.cls`, `acmart.cls`, `acm*.bbx/cbx/dbx`) are vendored here
from the JAIR Author Kit
(<https://www.jair.org/index.php/jair/libraryFiles/downloadPublic/6>).
