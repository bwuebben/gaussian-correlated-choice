# Primary sources and attribution boundaries

Verified 5 September 2026. This record supports the companion rather than claiming
an exhaustive history of Cotton's work or a complete priority search.

| Source | Verified material and use |
| --- | --- |
| Cotton, *Inferring Relative Ability from Winning Probability in Multientrant Contests*, SIAM J. Financial Math. 12(1), 295–317 (2021), DOI 10.1137/19M1276261 | Author PDF obtained from the microprediction/home repository. Translated-performance inversion, reusable competing field and lattice ties; Section 4.1 inquiry-cost application. Market shares are possible model inputs, not proof of a true demand system. |
| Cotton, *Scalable Inversion of Contests with Correlated Performances, Including Softmax and Multinomial Probit*, arXiv:2609.01133v1 (2026) | Primary arXiv HTML. Shared forward calculations conditional on factors and other structured dependence, derivatives and inversion. The paper fixes/represents a covariance for inversion; the companion does not attribute our experimental identification theorem to it. No independent replication of its large-scale speed claims is made. |
| Cotton, *Microprediction: Building an Open AI Network*, MIT Press (2022) | Publisher page confirms title, author, publication date 8 November 2022 and the network-of-autonomous-contributors motivation. We do not infer current employment from the historical publisher biography. |
| Cotton, *A Platform for Assessing and Combining Autonomous Short-Horizon Distributional Predictions* | Author PDF dated 7 April 2022, marked as a submission manuscript. Streams, autonomous crawlers, distributional forecasts, delayed eligibility, near-the-pin reward and combination. Cited as a manuscript, not as an accepted journal article. The PDF specifies a historical 225-scenario representation; no current live protocol parameter is asserted. |
| Cotton, *Schur Complementary Allocation: A Unification of Hierarchical Risk Parity and Minimum Variance Portfolios*, arXiv:2411.05807v1 (2024) | Exact title and abstract verified on arXiv. Cited only to distinguish Cotton's separate portfolio-construction work from contest inversion. |
| Abernethy, Chen and Vaughan (2013), ACM Trans. Econ. Comp. 1(2), Article 12, 39 pages, DOI 10.1145/2465769.2465777 | Author publication page, author-hosted preprint and Harvard-hosted published PDF. Convex cost-function market-making framework. Gaussian surplus example and elementary loss calculation are an application of that framework, not claimed new market-design theory. |
| Google Cloud, AP2 announcement, 16 September 2025 | Primary launch announcement. Used only for the distinction between transaction authorization/execution and statistical provider selection. No AP2 integration was performed. |
| Wuebben, *Gaussian Choice from Sparse Pairwise Interventions* (2026) | Local companion research manuscript. The controlled-anchor theorem is restated with hypotheses; local sampling and menu-stability limitations retained. |
| Hillis and Brainard, *Do Common Mechanisms of Adaptation Mediate Color Discrimination and Appearance? Uniform Backgrounds*, J. Opt. Soc. Am. A 22(10), 2090–2106 (2005) | Primary PDF hosted by NYU, read for the intervention revision. Its abstract and model discussion describe nonlinear response and adaptation with changes in luminance/chromatic context. Used to qualify physical-to-perceptual calibration, not as evidence that our Gaussian intervention assumptions hold. |

Primary URLs appear in `references.bib`. The URLs and SHA-256 hashes of
consulted author PDFs are retained in `source_downloads.json`; third-party
papers are not redistributed here. The ROAR and subset manuscripts in
that record were background reading and are not used for historical
dating or substantive claims. The ROAR copy has an anonymous byline
and is not treated as verified authorship or a dated publication.

All table and figure values in the companion are our synthetic calculations,
not numbers copied from Cotton's performance benchmarks. The existing allocation
example comes from the local mathematics manuscript and its saved source data.

## Geometry additions, checked 6 September 2026

| Source | Use and boundary |
| --- | --- |
| [Fosgerau and de Palma, *Generalized Entropy Models*](https://mpra.ub.uni-muenchen.de/70249/1/MPRA_paper_70249.pdf), Theorem 4 | Convex conjugacy of expected maximum utility on the simplex. This is background, not a new duality result. |
| [Müller, Nesterov and Shikhman, *Discrete Choice Prox-Functions on the Simplex*](https://arxiv.org/html/1909.05591), Section 2 | Surplus gradient, conjugate, signed cross-Hessian and zero row sums under the stated full-support assumptions. |
| [Cotton (2026)](https://arxiv.org/html/2609.01133v1), Section 5 and Proposition 1 | Tie-face boundary flux and structured probability derivatives. Cotton explicitly acknowledges the earlier cross-Hessian structure; his minimum-performance Jacobian has the opposite sign from the maximum-utility Hessian used here. |
| [Nielsen, *An Elementary Introduction to Information Geometry*](https://arxiv.org/abs/1808.08271), Entropy 22(10), 1100 (2020) | Classical Hessian dual coordinates, Bregman geometry and the categorical Fisher benchmark. The companion derives the scale factor and tangent-space quadratic form explicitly. |
| [Spielman and Srivastava, *Graph Sparsification by Effective Resistances*](https://arxiv.org/abs/0803.0929), Section 2 | Weighted graph Laplacian pseudoinverse and electrical resistance. No claim to new resistance theory or a graph-sparsification algorithm is made. |
| [Fiedler, *Algebraic Connectivity of Graphs*](https://dml.cz/handle/10338.dmlcz/101168), Section 1 | The classical second-eigenvalue notion. The companion uses its weighted Rayleigh-quotient analogue and distinguishes an absolute small gap from evidence of a relative bottleneck. |
| [Mertikopoulos and Sandholm, *Riemannian Game Dynamics*](https://users.ssc.wisc.edu/~whs/research/2018_JET.pdf), Sections 4 and 7 | Prior use of Hessian metrics and Bregman divergences on the simplex in economic dynamics; establishes adjacent geometric precedent, not an empirical routing result. |

The additions synthesize established mathematics for the stated contest model.
The Gaussian Hessian metric is not generally the Fisher metric of observed
categorical winners. The three-group explanation establishes sufficiency for
the given reconstruction, not impossibility of every two-group design.

`src/geometry.py` computes the additional table from the two existing synthetic
Gaussian inversions. Its Gaussian-conditioning formula is checked against
independent numerical derivatives of winner probabilities. It also checks
inverse and market scaling, the local Bregman expansion, the Gumbel scale
factor, and a nonsymmetric matrix example of the three-group identity.
Inputs, full matrices, spectra and numerical errors are in `output/geometry.json`.
