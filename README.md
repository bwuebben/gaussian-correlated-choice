# Gaussian Correlated Choice

Three papers by **Bernd Johannes Wuebben** on identifying correlated Gaussian
choice models, using them for decisions, and testing their commercial value.
This repository includes the manuscripts, LaTeX sources, numerical programs,
and the results used to produce their tables and figures.

## Papers

| Paper | Contribution | Files |
| --- | --- | --- |
| **Gaussian Choice from Sparse Pairwise Interventions** | Constructive identification from three groups of reference alternatives, local stability and sampling bounds, and synthetic experiments. | [PDF](paper/gaussian-choice-identification.pdf) · [LaTeX](paper/main.tex) |
| **From Winning Probabilities to Decisions: Prediction Markets, Agentic Commerce, and Portfolio Allocation** | Worked applications of contest inversion to ranked contracts, provider inquiry and substitution, and constrained allocation; identifies the limits of recovering portfolio risk from choices. | [PDF](companion/winning-probabilities-to-decisions.pdf) · [LaTeX](companion/main.tex) |
| **Commercial Value from Correlated Choice Models: Offer Optimization, Event Trading, and AI Service Routing** | Merchant experiments, event-contract trading research, and verified service routing, with unit economics, experimental requirements, and a staged validation plan. | [PDF](commercial/commercial-value-correlated-choice.pdf) · [LaTeX](commercial/main.tex) |

The mathematics paper is the starting point for the identification results.
The applications companion explains their uses and limitations through worked
examples. The commercial paper develops proposals that require further data
and prospective evaluation.

## Mathematical scope and evidence

Under a stable Gaussian random utility model with rank-$r$ factor covariance
and unknown diagonal noise, calibrated binary interventions involving three
affinely independent groups of $r+1$ reference alternatives identify the
centered mean and covariance. With $k=3(r+1)$, the construction uses
$kn-k(k+1)/2$ distinct pairs, $O(nr^2+r^3)$ arithmetic, and $O(nr)$ storage.
The sampling results require bounded parameters, response overlap, positive
idiosyncratic variance, and suitably conditioned reference groups.

Random-pair recovery with unknown diagonal variance, automatic selection of
reference alternatives, and global guarantees for the practical nonconvex
estimator remain open. Pairwise fit alone cannot establish stability across
larger menus or treatments.

**All numerical studies are synthetic.** The commercial tables are illustrative
planning calculations. No respondent study, customer pilot, measured commercial
uplift, executed trading strategy, or investment return is reported. Saved
results include seeds, source hashes, and optimizer diagnostics; failed or
incomplete numerical fits are retained. Numerical checks support reproducibility
and do not replace the mathematical arguments or external validation.

## Repository contents

| Path | Contents |
| --- | --- |
| `paper/` | Identification manuscript, bibliography, generated tables, and figures |
| `companion/` | Applications manuscript, numerical examples, and source notes |
| `commercial/` | Commercial manuscript, planning assumptions, calculations, and source notes |
| `src/` | Choice probabilities, estimation, sparse reconstruction, and simulation programs |
| `output/` | Saved synthetic experiments, validation reports, and aggregate results |
| `requirements.txt` | Pinned direct Python dependencies |
| `environment.lock.txt` | Package versions in the recorded numerical environment |

The [result summary](output/results_summary.json) records the inputs used for
the mathematics paper. The [companion examples](companion/output/examples.json)
and [commercial calculations](commercial/output/economics.json) record their
inputs and source hashes. The commercial assumptions are explicit in
[assumptions.json](commercial/assumptions.json).

## Build the PDFs

Install a TeX distribution with `latexmk`, AMS-LaTeX, and the packages used in
the sources, then run:

```bash
./build.sh
```

This builds all three papers using the included tables and figures. Each PDF
is written to the descriptive filename linked above. Temporary TeX files go
into an ignored `build/` directory inside each manuscript directory.
An individual source also builds with `latexmk -pdf main.tex` from its directory.

## Reproduce the calculations

The recorded environment is Python 3.14.6 with the versions in
`requirements.txt`; results on other versions may differ numerically.
No external dataset, account, credentials, or commercial API is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./reproduce.sh --saved
```

The `--saved` mode reruns population and estimation checks, verifies the source
hashes of saved simulation cases, regenerates all tables and figures, reruns
the companion and commercial examples, and builds all three PDFs. It is the
default mode. It uses the included simulation results for the larger studies.

To rerun the full configured experiment pipeline:

```bash
./reproduce.sh --full
```

The large experiment grids reuse cases only when their configuration, source
hashes, and recorded numerical versions match. To recompute every case, first
move `output/` to a backup location and then run `--full`. The complete run
includes population calculations through one million alternatives and can
require substantial time and memory. It does not construct a dense
million-alternative covariance or compute a million-way choice probability.
Timings depend on the machine; Monte Carlo and numerical integration can also
vary across platforms.

Use `PY=/absolute/path/to/python ./reproduce.sh --saved` to select another
environment. The script requests one numerical-library thread. Optimizer
convergence flags and integration tolerances are diagnostics, not certified
global error bounds. Six saved lifted-completion comparisons reached their
iteration limit and remain in the reported results.

The commercial calculation can be run independently using only the Python
standard library:

```bash
python3 commercial/src/economics.py
```

Changing commercial assumptions requires updating the corresponding discussion
as well as regenerating the tables. External interfaces and fee schedules cited
in the manuscripts are dated source descriptions and may subsequently change.

## Citation and license

Please cite the relevant paper when using its results. Bibliographic metadata
for the repository is provided in [CITATION.cff](CITATION.cff), and each paper
includes its own bibliography.

Code, numerical results, calculation inputs, and repository documentation are
available under the [MIT License](LICENSE). Manuscript sources, bibliographies,
typeset tables, figures, and PDFs are copyright © 2026 Bernd Johannes Wuebben;
all rights reserved pending journal publication.
