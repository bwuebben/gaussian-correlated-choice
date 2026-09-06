# Source and claim register

Primary sources checked 6 September 2026. This is targeted diligence for a
commercial research draft, not an exhaustive competitor, legal or novelty review.
No customer interviews, proprietary datasets, live trades or replicated external
backtests are represented by these sources.

| Key | Primary source | Supported use and boundary |
| --- | --- | --- |
| cotton2021 | [SIAM article](https://doi.org/10.1137/19M1276261); author-PDF provenance retained in `../companion/SOURCES.md` and `../companion/source_downloads.json` | Contest inversion and dealer inquiry with inquiry cost. The companion and its retained primary-source record document the use of the 2021 article. No current software performance or adoption claim. |
| cotton2026 | [arXiv v1](https://arxiv.org/html/2609.01133v1) | Structured correlated contests, shared computation, inversion and derivatives conditional on a modeled shock law. No claim that a trade identifies the latent mean update. |
| wuebbenident | [Mathematics manuscript](../paper/main.tex) | Three-group sparse identification; exact rank, calibration, stability, conditioning and repeated-sampling boundaries. No external empirical validation. |
| wuebbencompanion | `../companion/main.tex`, 5 September revision | Comparative versus absolute information, inquiry example, latent/payoff covariance distinction, and portfolio limits. Its examples are synthetic. |
| intelligems | [Vendor listing](https://apps.shopify.com/intelligems) | Existing price, offer and checkout testing with COGS-based metrics. No reported vendor uplift is adopted, and no claim to exhaustive competitor coverage. |
| ucp | [Google developer announcement, 11 January 2026](https://developers.googleblog.com/under-the-hood-universal-commerce-protocol-ucp/) | Commerce protocol, extensions and discount example. Does not establish access to buyer consideration sets or a merchant auction. |
| directoffers | [Google product announcement](https://blog.google/products/ads-commerce/google-marketing-live-search-ads/) | Expansion of the Direct Offers pilot described in May 2026. Does not imply public availability to the proposed company. |
| acp | [OpenAI announcement, 29 September 2025](https://openai.com/index/buy-it-in-chatgpt/) | Historical launch of Instant Checkout and ACP with Stripe. Do not present the launch's Shopify plans as the current channel architecture. |
| shopifycommerce | [Shopify guide, updated 18 June 2026](https://www.shopify.com/blog/how-agentic-commerce-works) | Describes channel-specific paths including merchant-site checkout in an in-app browser for ChatGPT. Vendor growth figures are not used as a market estimate. Actual pilot capabilities require confirmation. |
| dudik2011 | [Original authors' paper](https://arxiv.org/abs/1103.4601) | Standard doubly robust policy-evaluation reference. The estimator and its assumptions are printed in our draft; no methodological priority claim. |
| saguillo2025 | [AFT 2025 proceedings](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.AFT.2025.27) and [primary PDF](https://drops.dagstuhl.de/storage/00lipics/lipics-vol354-aft2025/LIPIcs.AFT.2025.27/LIPIcs.AFT.2025.27.pdf) | Prior empirical analysis of arbitrage and execution, not current accessible profit. The published study's April 2024–April 2025 period and non-atomic execution discussion were inspected. No external profit number is used in our projections. |
| kalshiapi / kalshidata | [Introduction](https://docs.kalshi.com/welcome), [public data guide](https://docs.kalshi.com/getting_started/quick_start_market_data) | Documented market-data/execution interfaces and public REST quick start. No authenticated access or successful market-data collection is claimed. |
| kalshifees | [Primary fee PDF](https://kalshi.com/docs/kalshi-fee-schedule.pdf) | Retrieved document says effective 7 July 2026. General taker coefficient 0.07, multipliers, exceptions and maker fees. Its rounding text and illustrative table appear inconsistent in places; the paper only calculates the unrounded formula and requires production reconciliation. |
| kalshihistory | [Historical data docs](https://docs.kalshi.com/getting_started/historical_data) | Live/historical split and endpoint types. Does not prove complete historical order-book depth or queue reconstruction. |
| kalshibook | [Order-book endpoint](https://docs.kalshi.com/api-reference/market/get-market-orderbook) | Current bid-side quantities and YES/NO complement convention. The public quick start and generated authenticated endpoint samples differ in presentation; no credential requirement is inferred beyond the respective documentation. |
| polymarketfees | [Fee documentation](https://docs.polymarket.com/trading/fees) | Category-dependent parameters. No universal fee-free assumption, no geographic-access claim, and no wallet or trading action. |
| routellm | [RouteLLM, arXiv:2406.18665](https://arxiv.org/abs/2406.18665) | Existing preference-based model routing; version history checked. Its benchmark savings are not treated as our commercial outcome. |
| bedrock | [AWS developer documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-routing.html) | Existing commercial prompt-routing product. No assertion of an unoccupied routing market. |
| agentreputation | [arXiv:2605.00073](https://arxiv.org/abs/2605.00073) | Authors and submission date checked against the arXiv record. Abstract discusses context transfer, evaluation and incentives; no marketplace adoption, loss underwriting, or empirical reliability is inferred. |

## Numerical provenance

`assumptions.json` contains every input for the seven generated tables. All are
planning assumptions or elementary constructed examples. `src/economics.py`
writes `output/economics.json` with input/source SHA-256 hashes and generates the
LaTeX tables. No vendor result, observed probability, customer forecast, or
backtest is hidden inside those inputs.

The current external sources establish infrastructure and related work only.
The remaining missing evidence is named explicitly: customer access and paid
pilot, causal contribution uplift, routing total-cost improvement, executable
trading edge, capacity, and legal/contractual deployment conditions.
