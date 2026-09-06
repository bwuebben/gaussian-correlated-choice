#!/usr/bin/env python3
"""Reproduce planning arithmetic. No market data, fitted returns, or API calls.

Inputs are deliberately labelled assumptions. Generates JSON and LaTeX tables
from the same values, with independent identities checked before writing.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]


def dollars(value: float, decimals: int = 0) -> str:
    return f"\\${value:,.{decimals}f}"


def table(columns: str, header: str, rows: list[str]) -> str:
    return "\n".join([
        f"\\begin{{tabular}}{{{columns}}}", "\\toprule", header + r" \\",
        "\\midrule", *[row + r" \\" for row in rows], "\\bottomrule",
        "\\end{tabular}", "",
    ])


def compute(a: dict) -> dict:
    m, t, p, r = (a[k] for k in ("merchant", "trading", "power", "routing"))
    offers = []
    for row in m["offer_scenarios"]:
        assert math.isclose(row["p_A"] + row["p_B"] + row["p_outside"], 1)
        ma = m["contribution_A_before_discount"] - row["discount_A"]
        mb = m["contribution_B"]
        expected = row["p_A"] * ma + row["p_B"] * mb
        variance = row["p_A"] * ma**2 + row["p_B"] * mb**2 - expected**2
        offers.append({**row, "conversion": row["p_A"] + row["p_B"],
                       "contribution_per_session": expected, "variance": variance})
    base = offers[0]["contribution_per_session"]
    for row in offers:
        row["increment_per_session"] = row["contribution_per_session"] - base
    assert offers[2]["conversion"] > offers[0]["conversion"]
    assert offers[2]["contribution_per_session"] < base
    assert max(offers, key=lambda x: x["contribution_per_session"])["discount_A"] == 5
    merchants = []
    for uplift in m["uplift_rates"]:
        revenue = m["eligible_annual_baseline_contribution"] * uplift * m["performance_fee_rate"]
        unit_contribution = revenue - m["annual_direct_cost_per_customer"]
        merchants.append({"uplift": uplift, "revenue_per_customer": revenue,
                          "direct_contribution_per_customer": unit_contribution,
                          "customers_for_break_even": math.ceil(m["annual_fixed_operating_cost"] / unit_contribution),
                          "customers_for_target_profit": math.ceil((m["annual_fixed_operating_cost"] + m["target_operating_profit"]) / unit_contribution),
                          "scales": [{"customers": n, "revenue": n * revenue,
                                      "operating_profit": n * unit_contribution - m["annual_fixed_operating_cost"]}
                                     for n in m["customer_counts"]]})
    trading = []
    for edge in t["net_edge_per_completed_unit"]:
        required = (t["target_operating_profit"] + t["annual_fixed_operating_cost"]) / edge
        trading.append({"net_edge": edge, "required_completed_units": required,
                        "required_addressable_units": required / t["participation_rate"],
                        "illustrative_cash_capacity": required / t["annual_turns"] * t["cash_tied_per_unit"]})
    entry = t["example_yes_ask"] + t["example_no_ask"]
    fee = sum(t["illustrative_taker_coefficient"] * x * (1-x)
              for x in (t["example_yes_ask"], t["example_no_ask"]))
    execution_expectation = ((1-t["example_failure_probability"]) * t["example_complete_gain"]
                             - t["example_failure_probability"] * t["example_failed_hedge_loss"])
    assert entry < 1 and 1-entry-fee < 0
    nd = NormalDist()
    zsum = nd.inv_cdf(1-p["two_sided_alpha"]/2) + nd.inv_cdf(p["power"])
    power = [{"effect": effect,
              "per_arm": math.ceil(2 * zsum**2 * p["session_profit_sd"]**2 / effect**2)}
             for effect in p["absolute_effects"]]
    routing = []
    fa, fb = r["first_failure_probability"], r["second_marginal_failure_probability"]
    for joint in r["joint_failure_probabilities"]:
        assert max(0, fa+fb-1) <= joint <= min(fa, fb)
        routing.append({"joint_failure": joint, "fallback_conditional_failure": joint/fa,
                        "expected_total_cost": r["first_attempt_cost"] + fa * r["second_attempt_cost"] + joint * r["terminal_remediation_cost"]})
    reliability = []
    for target in a["reliability"]["failure_probability_targets"]:
        alpha = a["reliability"]["one_sided_alpha"]
        n = math.ceil(math.log(alpha) / math.log1p(-target))
        assert 1-alpha**(1/n) <= target
        assert 1-alpha**(1/(n-1)) > target
        reliability.append({"target": target, "zero_failure_trials": n})
    anchors = []
    k = 3 * (a["anchor_examples"]["r"] + 1)
    for n in a["anchor_examples"]["n"]:
        count = k*n-k*(k+1)//2
        assert count == k*(n-k)+k*(k-1)//2
        anchors.append({"n": n, "pairs": count,
                        "trials": 2 * count * a["anchor_examples"]["trials_per_level"]})
    # Exact implication bundle: buy 1_B plus 1-1_A when A is a subset of B.
    implication_payoffs = [b+1-a for a, b in [(0, 0), (0, 1), (1, 1)]]
    assert min(implication_payoffs) == 1
    assert sum(a["milestone_tranches"]) == sum(a["six_month_budget"].values())
    central = next(x for x in merchants if x["uplift"] == 0.03)
    eligible_sales = m["eligible_annual_baseline_contribution"] / m["contribution_margin"]
    institution = a["institutional_inquiry"]
    inquiry_savings = institution["annual_routed_notional"] * institution["execution_improvement_bps"] / 10000
    narrative = {"eligible_annual_sales": eligible_sales,
                 "total_annual_sales_if_restricted": eligible_sales / m["eligible_sales_fraction"],
                 "acquisition_payback_months": 12*m["acquisition_and_onboarding_cash"]/central["direct_contribution_per_customer"],
                 "accounts_for_revenue_targets": {str(target): math.ceil(target/central["revenue_per_customer"])
                                                  for target in m["recurring_revenue_targets"]},
                 "institutional_inquiry_savings": inquiry_savings,
                 "institutional_inquiry_fee": inquiry_savings * institution["fee_share"]}
    return {"classification": a["classification"], "offers": offers,
            "merchants": merchants, "trading": trading,
            "fee_example": {"gross_profit": 1-entry, "unrounded_fees": fee,
                            "profit_after_unrounded_fees": 1-entry-fee},
            "execution_example_expected_profit": execution_expectation,
            "power": power, "routing": routing,
            "routing_annual_revenue_per_customer": r["annual_tasks_per_customer"] * r["verified_saving_per_task"] * r["value_share"],
            "reliability": reliability, "anchors": anchors,
            "six_month_budget": sum(a["six_month_budget"].values()),
            "implication_payoffs": implication_payoffs, "narrative_calculations": narrative}


def main() -> None:
    assumption_path = ROOT / "assumptions.json"
    a = json.loads(assumption_path.read_text())
    out = compute(a)
    out["provenance"] = {"assumptions_sha256": hashlib.sha256(assumption_path.read_bytes()).hexdigest(),
                         "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    generated = ROOT / "generated"
    generated.mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output/economics.json").write_text(json.dumps(out, indent=2) + "\n")
    (generated / "offers.tex").write_text(table("rrrrrr", r"Discount A & $p_A$ & $p_B$ & No purchase & Conversion & Contribution",
        [f"{dollars(x['discount_A'])} & {x['p_A']:.3f} & {x['p_B']:.3f} & {x['p_outside']:.3f} & {100*x['conversion']:.1f}\\% & {dollars(x['contribution_per_session'],3)}" for x in out["offers"]]))
    (generated / "merchant.tex").write_text(table("rrrrr", r"Uplift & Annual fee & Direct contribution & Break-even clients & Clients for \$1m",
        [f"{100*x['uplift']:.0f}\\% & {dollars(x['revenue_per_customer'])} & {dollars(x['direct_contribution_per_customer'])} & {x['customers_for_break_even']:,} & {x['customers_for_target_profit']:,}" for x in out["merchants"]]))
    (generated / "trading.tex").write_text(table("rrrr", r"Net edge & Completed units/year & Addressable units/year & Cash capacity",
        [f"{dollars(x['net_edge'],3)} & {x['required_completed_units']/1e6:,.0f}m & {x['required_addressable_units']/1e9:,.0f}bn & {dollars(x['illustrative_cash_capacity'])}" for x in out["trading"]]))
    (generated / "power.tex").write_text(table("rrr", r"Effect/session & Observations/arm & Total observations",
        [f"{dollars(x['effect'],3)} & {x['per_arm']:,} & {2*x['per_arm']:,}" for x in out["power"]]))
    (generated / "routing.tex").write_text(table("rrr", r"Joint failure & Failure of B given A failed & Expected cost/task",
        [f"{100*x['joint_failure']:.0f}\\% & {100*x['fallback_conditional_failure']:.0f}\\% & {dollars(x['expected_total_cost'],2)}" for x in out["routing"]]))
    (generated / "anchors.tex").write_text(table("rrr", r"Alternatives & Distinct pairs & Binary observations ($N=100$)",
        [f"{x['n']:,} & {x['pairs']:,} & {x['trials']:,}" for x in out["anchors"]]))
    (generated / "budget.tex").write_text(table("lr", r"Six-month expense assumption & Amount",
        [f"{k.replace('_', ' ').capitalize()} & {dollars(v)}" for k,v in a["six_month_budget"].items()] + [f"Total & {dollars(out['six_month_budget'])}"]))
    print(json.dumps({"checks": "passed", "classification": out["classification"],
                      "budget": out["six_month_budget"], "fee_example": out["fee_example"],
                      "power": out["power"], "reliability": out["reliability"]}, indent=2))


if __name__ == "__main__":
    main()
