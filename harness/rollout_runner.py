"""One-call benchmark evaluation: rollout transcripts plus optional judging."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.pipeline.client import build_client, is_local_model, load_dotenv

from .clients import build_eval_client
from .config import EvalConfig, EvalReport
from .prices import cost_usd
from .prompts import resolve_assistant_prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_run_dir_samples(source: Path) -> list[dict[str, Any]]:
    path = source / "samples.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _apply_axis_patch(axis: str) -> None:
    if axis == "sycophancy":
        from archetype_guided_benchmark.bootstrap import (
            apply_archetype_prompt_patch,
        )
        from archetype_guided_benchmark.prompts import (
            configure_species_lock_variant,
        )

        configure_species_lock_variant()
        apply_archetype_prompt_patch()
    else:
        from archetype_guided_benchmark.bootstrap_validation_v28 import (
            apply_validation_v28_archetype_prompt_patch,
        )

        apply_validation_v28_archetype_prompt_patch()


def _rollout(
    *,
    sample: dict[str, Any],
    assistant_client: Any,
    user_client: Any,
    assistant_system_prompt: str,
    assistant_max_tokens: int | None = 4000,
) -> dict[str, Any]:
    from rule_guided_benchmark import prompts as rg_prompts

    scenario = sample["scenario"]
    texting_style = sample.get("texting_style") or {}
    assistant_client = copy.copy(assistant_client)
    assistant_client.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                              "cached_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    assistant_client._usage_lock = threading.Lock()
    user_client = copy.copy(user_client)
    user_client.usage = {"prompt_tokens": 0, "completion_tokens": 0,
                         "cached_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    user_client._usage_lock = threading.Lock()
    history: list[dict[str, str]] = []
    for user_turn in range(1, 6):
        user_message = user_client.complete(
            system=rg_prompts.build_roleplayer_prompt(
                scenario, texting_style, user_turn, history
            ),
            messages=[
                {
                    "role": "user",
                    "content": "Write the next in-character user message only.",
                }
            ],
            temperature=0.85,
            max_tokens=1200,
        ).strip()
        if not user_message:
            raise ValueError(f"{sample['id']}: user model returned empty text")
        history.append({"role": "user", "content": user_message})
        assistant_message = assistant_client.complete(
            system=assistant_system_prompt,
            messages=[
                {"role": item["role"], "content": item["content"]}
                for item in history
            ],
            temperature=0.25,
            max_tokens=assistant_max_tokens,
            # Final user turn is never re-read: skip its cache breakpoint
            # to avoid a pure-waste cache write (writes bill ~1.25x input).
            cache_tail=(user_turn < 5),
        ).strip()
        if not assistant_message:
            raise ValueError(f"{sample['id']}: model under test returned empty text")
        history.append({"role": "assistant", "content": assistant_message})
    assistant_usage = dict(assistant_client.usage)
    user_usage = dict(user_client.usage)
    assistant_cost = cost_usd(
        assistant_client.model,
        assistant_usage,
        getattr(assistant_client, "service_tier", None),
    )
    user_cost = cost_usd(user_client.model, user_usage)
    return {
        "schema_version": 2,
        "scenario_id": sample["id"],
        "benchmark_axis": scenario.get("benchmark_axis"),
        "assistant_model": assistant_client.model,
        "user_model": user_client.model,
        "usage": {"assistant": assistant_usage, "user": user_usage},
        "cost_usd": {"assistant": assistant_cost, "user": user_cost},
        "turns": [
            {"turn": index + 1, "role": item["role"], "content": item["content"]}
            for index, item in enumerate(history)
        ],
    }


def _log_rollout_cost(sample_id: str, transcript: dict[str, Any]) -> None:
    cost = transcript.get("cost_usd") or {}
    usage = transcript.get("usage") or {}
    a = usage.get("assistant", {})
    price = cost.get("assistant")
    print(
        f"EVAL {sample_id} done "
        f"asst_cost=${price if price is not None else 'n/a'} "
        f"(in={a.get('prompt_tokens', 0)} cached={a.get('cached_tokens', 0)} "
        f"out={a.get('completion_tokens', 0)} think={a.get('reasoning_tokens', 0)})",
        flush=True,
    )


def _sum_costs(out_dir: Path) -> dict[str, Any]:
    total: dict[str, Any] = {"assistant": 0.0, "user": 0.0, "unknown_price": 0}
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
             "reasoning_tokens": 0, "calls": 0}
    n = 0
    for path in sorted((out_dir / "transcripts").glob("*.json")):
        try:
            t = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        c = t.get("cost_usd") or {}
        u = (t.get("usage") or {}).get("assistant", {})
        if c.get("assistant") is None:
            total["unknown_price"] += 1
        else:
            total["assistant"] += c["assistant"]
        if c.get("user") is not None:
            total["user"] += c["user"]
        for k in usage:
            usage[k] += int(u.get(k, 0) or 0)
        n += 1
    total["assistant"] = round(total["assistant"], 4)
    total["user"] = round(total["user"], 4)
    return {"n_transcripts": n, "cost_usd": total, "assistant_usage": usage}


def evaluate(config: EvalConfig) -> EvalReport:
    """Run an evaluation sweep; return paths plus per-sample outcomes."""
    config.validate()
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    samples = (
        list(config.samples)
        if isinstance(config.samples, list)
        else _load_run_dir_samples(Path(config.samples))
    )
    if config.sample_ids:
        wanted = set(config.sample_ids)
        samples = [s for s in samples if s["id"] in wanted]
        missing = wanted - {s["id"] for s in samples}
        if missing:
            raise ValueError(f"unknown sample ids: {sorted(missing)}")
    if config.limit:
        samples = samples[: config.limit]
    if not samples:
        raise ValueError("no samples selected")
    from .ground_truth import assert_final_samples

    assert_final_samples(samples)

    out_dir = Path(config.output_dir)
    transcripts_dir = out_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(config.to_json() + "\n", encoding="utf-8")

    assistant_system_prompt = resolve_assistant_prompt(config.assistant_prompt)
    assistant_client = build_eval_client(
        model=config.model_under_test,
        api_key=api_key,
        endpoint_url=config.endpoint_url,
        served_model=config.served_model,
        endpoint_api_key=config.endpoint_api_key,
        reasoning_effort=config.reasoning_effort,
        timeout=config.timeout,
        retries=config.retries,
        service_tier=config.service_tier,
    )
    user_client = build_client(
        config.user_model,
        api_key=api_key,
        timeout=config.timeout,
        retries=config.retries,
        **({} if is_local_model(config.user_model) else {
            "reasoning_effort": config.user_reasoning_effort,
            "service_tier": config.user_service_tier,
        }),
    )

    pending = [
        s
        for s in samples
        if not (transcripts_dir / f"{s['id']}.json").is_file()
    ]
    report = EvalReport(output_dir=out_dir)

    def axis_of(sample: dict[str, Any]) -> str:
        return str(
            sample.get("benchmark_axis")
            or sample.get("scenario", {}).get("benchmark_axis")
            or "sycophancy"
        )

    for axis in ("sycophancy", "calibrated_validation"):
        group = [s for s in pending if axis_of(s) == axis]
        if not group:
            continue
        _apply_axis_patch(axis)
        with ThreadPoolExecutor(
            max_workers=min(config.workers, len(group))
        ) as executor:
            futures = {
                executor.submit(
                    _rollout,
                    sample=sample,
                    assistant_client=assistant_client,
                    user_client=user_client,
                    assistant_system_prompt=assistant_system_prompt,
                    assistant_max_tokens=config.assistant_max_tokens,
                ): sample
                for sample in group
            }
            for future in as_completed(futures):
                sample = futures[future]
                try:
                    transcript = future.result()
                except Exception as exc:
                    report.failed[sample["id"]] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    print(f"EVAL {sample['id']} FAILED: {exc}", flush=True)
                    continue
                (transcripts_dir / f"{sample['id']}.json").write_text(
                    json.dumps(transcript, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                report.completed.append(sample["id"])
                _log_rollout_cost(sample["id"], transcript)

    for sample in samples:
        path = transcripts_dir / f"{sample['id']}.json"
        if path.is_file() and sample["id"] not in report.completed:
            report.completed.append(sample["id"])

    judged_samples = []
    for sample in samples:
        path = transcripts_dir / f"{sample['id']}.json"
        if not path.is_file():
            continue
        entry = deepcopy(sample)
        entry["transcript"] = json.loads(path.read_text(encoding="utf-8"))
        entry["models"] = {
            "assistant": assistant_client.model,
            "user": user_client.model,
        }
        judged_samples.append(entry)
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for entry in judged_samples:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "eval": True,
                "publishable": False,
                "website_import": False,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    if config.judge != "skip" and judged_samples:
        from archetype_guided_benchmark.judge_v9 import DEFAULT_JUDGE_MODEL
        from rule_guided_judging.hr_v11_judge import (
            API_JUDGE_MODEL,
            run_locked_judge,
        )

        judge_model = (
            API_JUDGE_MODEL if config.judge == "api" else DEFAULT_JUDGE_MODEL
        )
        judge_run_id = f"{out_dir.name}-judged"
        manifest = run_locked_judge(
            api_key=api_key,
            source_run=out_dir,
            judge_run_id=judge_run_id,
            judge_model=judge_model,
            workers=min(24, len(judged_samples)),
        )
        report.judge_run_id = judge_run_id
        report.judge_counts = manifest.get("counts")

    totals = _sum_costs(out_dir)
    print(f"EVAL TOTAL {out_dir.name}: {totals}", flush=True)
    (out_dir / "summary.json").write_text(
        json.dumps(
            {
                "completed": sorted(report.completed),
                "failed": report.failed,
                "judge_run_id": report.judge_run_id,
                "judge_counts": report.judge_counts,
                "cost_totals": totals,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report
