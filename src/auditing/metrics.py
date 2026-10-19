"""Metrics shared by all verifier auditing strategies."""


def compute_audit_rate_metrics(
    *, candidates: float, audited: float, batch_size: float
) -> dict[str, float]:
    """Compute response-level audit rates, omitting undefined ratios."""
    metrics = {"audit_batch_size": batch_size}
    if batch_size > 0:
        metrics["audit_candidate_prevalence"] = candidates / batch_size
        metrics["audit_rate"] = audited / batch_size
    if candidates > 0:
        metrics["audit_candidate_coverage"] = audited / candidates
    return metrics


def compute_audit_metrics(
    *,
    candidates: float,
    audited: float,
    batch_size: float,
    audited_false_positives: float,
    false_positives_before: float,
    false_positives_after: float,
    audit_budget: float,
    cumulative_audited: float,
    cumulative_audited_false_positives: float,
    exploration_count: float,
    exploitation_count: float,
) -> dict[str, float]:
    """Build comparable audit metrics while omitting undefined ratios."""
    metrics = {
        "audit_candidates": candidates,
        "audited": audited,
        **compute_audit_rate_metrics(
            candidates=candidates,
            audited=audited,
            batch_size=batch_size,
        ),
        "audited_false_positives": audited_false_positives,
        "false_positives_before_audit": false_positives_before,
        "false_positives_after_audit": false_positives_after,
        "cumulative_audited": cumulative_audited,
        "cumulative_audited_false_positives": (cumulative_audited_false_positives),
        "audit_exploration_count": exploration_count,
        "audit_exploitation_count": exploitation_count,
    }
    if audited > 0:
        metrics["audit_precision"] = audited_false_positives / audited
    if audit_budget > 0:
        metrics["audit_budget_utilization"] = audited / audit_budget
    if false_positives_before > 0:
        metrics["audit_false_positive_recall"] = (
            audited_false_positives / false_positives_before
        )
    positive_rewards_after_audit = candidates - audited_false_positives
    if positive_rewards_after_audit > 0:
        metrics["post_audit_reward_contamination"] = (
            false_positives_after / positive_rewards_after_audit
        )
    if batch_size > 0:
        metrics["reward_correction_rate"] = audited_false_positives / batch_size
    if cumulative_audited > 0:
        metrics["cumulative_audit_precision"] = (
            cumulative_audited_false_positives / cumulative_audited
        )
    return metrics
