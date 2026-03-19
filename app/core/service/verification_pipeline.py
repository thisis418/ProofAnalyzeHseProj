import logging
from typing import Any, Callable

from app.core.agents.critic_agent import CriticAgent
from app.core.agents.formulator_agent import FormulatorAgent
from app.core.utils.proof_utils import ProofUtils

logger = logging.getLogger(__name__)


class VerificationPipeline:
    DEFAULT_MAX_ROUNDS_PER_PHASE: int = 1

    def __init__(
        self,
        formulator: FormulatorAgent,
        critic: CriticAgent,
        proof_utils: ProofUtils,
        max_rounds_per_phase: int | None = None,
    ) -> None:
        self.formulator = formulator
        self.critic = critic
        self.proof_utils = proof_utils
        self.max_rounds_per_phase = (
            max_rounds_per_phase or self.DEFAULT_MAX_ROUNDS_PER_PHASE
        )

    @staticmethod
    def _collect_final_remarks(debate_history: list[dict], phase: str) -> list[dict]:
        formulator_entries = [
            e
            for e in debate_history
            if e.get("agent") == "formulator" and e.get("phase") == phase
        ]
        critic_entries = [
            e
            for e in debate_history
            if e.get("agent") == "critic" and e.get("phase") == phase
        ]
        if not formulator_entries:
            return []
        last_f = formulator_entries[-1]
        last_c = critic_entries[-1] if critic_entries else {}
        remarks: list[dict] = []
        for step_id, step_data in last_f.get("step_results", {}).items():
            for r in step_data.get("remarks", []):
                remarks.append({**r, "reported_by": "formulator", "phase": phase})
        for gr in last_f.get("global_remarks", []):
            remarks.append({**gr, "reported_by": "formulator", "phase": phase})
        for step_id, step_data in last_c.get("step_reviews", {}).items():
            if not step_data.get("agrees_with_formulator", True):
                for r in step_data.get("critic_remarks", []):
                    remarks.append(
                        {
                            **r,
                            "step_id": int(step_id),
                            "reported_by": "critic",
                            "phase": phase,
                        }
                    )
        global_review = last_c.get("global_review", {})
        if not global_review.get("agrees", True):
            for gr in global_review.get("critic_global_remarks", []):
                remarks.append({**gr, "reported_by": "critic", "phase": phase})
        return remarks

    @staticmethod
    def _latest_phase_entry(
        debate_history: list[dict], agent: str, phase: str
    ) -> dict[str, Any]:
        for entry in reversed(debate_history):
            if entry.get("agent") == agent and entry.get("phase") == phase:
                return entry
        return {}

    @staticmethod
    def _unique_strings(values: list[Any]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value is None:
                continue
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result

    def _attach_used_theorems(
        self,
        proof_steps: list[dict[str, Any]],
        debate_history: list[dict],
        remarks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        enriched_steps = [dict(step) for step in proof_steps]
        per_step: dict[int, list[dict[str, Any]]] = {
            index: [] for index, _ in enumerate(enriched_steps)
        }
        formulator_facts = self._latest_phase_entry(
            debate_history, agent="formulator", phase="fact_checking"
        )
        critic_facts = self._latest_phase_entry(
            debate_history, agent="critic", phase="fact_checking"
        )

        for raw_step_id, step_data in formulator_facts.get("step_results", {}).items():
            try:
                step_id = int(raw_step_id)
            except (TypeError, ValueError):
                continue
            if step_id not in per_step:
                continue
            for fact in step_data.get("facts_found", []):
                references = self._unique_strings(
                    [fact.get("matched_theorem"), *(fact.get("rag_references", []))]
                )
                if not references and not fact.get("name"):
                    continue
                per_step[step_id].append(
                    {
                        "fact_name": fact.get("name"),
                        "theorem_name": fact.get("matched_theorem")
                        or (references[0] if references else fact.get("name")),
                        "matched_theorem": fact.get("matched_theorem"),
                        "references": references,
                        "sources": ["formulator"],
                    }
                )
            for remark in step_data.get("remarks", []):
                references = self._unique_strings(
                    [remark.get("fact_name"), *(remark.get("rag_references", []))]
                )
                if not references and not remark.get("fact_name"):
                    continue
                per_step[step_id].append(
                    {
                        "fact_name": remark.get("fact_name"),
                        "theorem_name": remark.get("fact_name")
                        or (references[0] if references else ""),
                        "matched_theorem": None,
                        "references": references,
                        "sources": ["formulator_remark"],
                    }
                )

        for raw_step_id, step_data in critic_facts.get("step_reviews", {}).items():
            try:
                step_id = int(raw_step_id)
            except (TypeError, ValueError):
                continue
            if step_id not in per_step:
                continue
            references = self._unique_strings(step_data.get("rag_references_found", []))
            if not references:
                continue
            per_step[step_id].append(
                {
                    "fact_name": None,
                    "theorem_name": references[0],
                    "matched_theorem": None,
                    "references": references,
                    "sources": ["critic"],
                }
            )

        for remark in remarks:
            step_id = remark.get("step_id")
            if not isinstance(step_id, int) or step_id not in per_step:
                continue
            references = self._unique_strings(
                [remark.get("fact_name"), *(remark.get("rag_references", []))]
            )
            if not references and not remark.get("fact_name"):
                continue
            per_step[step_id].append(
                {
                    "fact_name": remark.get("fact_name"),
                    "theorem_name": remark.get("fact_name")
                    or (references[0] if references else ""),
                    "matched_theorem": None,
                    "references": references,
                    "sources": ["final_remark"],
                }
            )

        for step_id, step in enumerate(enriched_steps):
            grouped: dict[str, dict[str, Any]] = {}
            for item in per_step[step_id]:
                references = self._unique_strings(item.get("references", []))
                key = "|".join(references) or str(item.get("fact_name") or "")
                if not key:
                    continue
                existing = grouped.get(key)
                if existing:
                    existing["references"] = self._unique_strings(
                        [*existing.get("references", []), *references]
                    )
                    existing["sources"] = self._unique_strings(
                        [*existing.get("sources", []), *item.get("sources", [])]
                    )
                    existing["fact_name"] = (
                        existing.get("fact_name") or item.get("fact_name")
                    )
                    existing["matched_theorem"] = (
                        existing.get("matched_theorem") or item.get("matched_theorem")
                    )
                    existing["theorem_name"] = (
                        existing.get("theorem_name") or item.get("theorem_name")
                    )
                    continue
                grouped[key] = {
                    "fact_name": item.get("fact_name"),
                    "theorem_name": item.get("theorem_name"),
                    "matched_theorem": item.get("matched_theorem"),
                    "references": references,
                    "sources": self._unique_strings(item.get("sources", [])),
                }
            step["used_theorems"] = [
                {
                    **item,
                    "theorem_name": item.get("theorem_name")
                    or next(
                        (
                            candidate
                            for candidate in [
                                *(item.get("references") or []),
                                item.get("fact_name"),
                            ]
                            if candidate
                        ),
                        "",
                    ),
                }
                for item in grouped.values()
            ]
        return enriched_steps

    async def _run_phase(
        self,
        phase_name: str,
        proof_steps: list[dict[str, Any]],
        debate_history: list[dict],
        context: dict[str, Any],
        formulator_fn: Callable,
        critic_fn: Callable,
        max_rounds: int,
    ) -> dict[str, Any]:
        logger.info("\n%s\n▶ Фаза: %s\n%s", "=" * 60, phase_name.upper(), "=" * 60)
        last_formulator_result: dict[str, Any] | None = None
        last_critic_result: dict[str, Any] | None = None
        for round_num in range(1, max_rounds + 1):
            logger.info("\n  📍 %s | Раунд %s/%s", phase_name, round_num, max_rounds)
            formulator_result = await formulator_fn(
                proof_steps, debate_history, context
            )
            formulator_result["round"] = round_num
            debate_history.append(formulator_result)
            last_formulator_result = formulator_result
            critic_result = await critic_fn(
                proof_steps, formulator_result, debate_history, context
            )
            critic_result["round"] = round_num
            debate_history.append(critic_result)
            last_critic_result = critic_result
            if critic_result.get("consensus_reached", False):
                logger.info("  ✅ Консенсус на раунде %s", round_num)
                break
            if round_num < max_rounds:
                logger.info("  ↩ Следующий раунд...")
            else:
                logger.warning("  ⚠ Лимит раундов исчерпан")
        return {
            "consensus_reached": last_critic_result.get("consensus_reached", False)
            if last_critic_result
            else False,
            "rounds_taken": max_rounds
            if last_critic_result and not last_critic_result.get("consensus_reached")
            else next(
                (
                    e.get("round", max_rounds)
                    for e in reversed(debate_history)
                    if e.get("phase") == phase_name and e.get("agent") == "critic"
                ),
                max_rounds,
            ),
            "last_formulator_result": last_formulator_result,
            "last_critic_result": last_critic_result,
            "phase": phase_name,
        }

    async def verify_proof(
        self,
        proof_id: str,
        latex: str,
        context: dict[str, Any] | None = None,
        max_rounds_per_phase: int | None = None,
    ) -> dict[str, Any]:
        context = context or {}
        max_rounds = max_rounds_per_phase or self.max_rounds_per_phase
        debate_history: list[dict] = []
        logger.info("\n%s", "#" * 60)
        logger.info("# Верификация: %s", proof_id)
        logger.info(
            "# Тема: %s, уровень: %s",
            context.get("topic", "—"),
            context.get("level", "—"),
        )
        logger.info("%s", "#" * 60)
        logger.info("▶ Парсинг LaTeX доказательства...")
        proof_steps = await self.proof_utils.parse_latex_proof(latex, context)
        logger.info("  Выделено шагов: %s", len(proof_steps))
        if not proof_steps:
            return {
                "proof_id": proof_id,
                "is_valid": False,
                "confidence_score": 0.0,
                "summary": "❌ Не удалось выделить шаги из доказательства",
                "iteration_recommendation": "Проверьте формат входного LaTeX",
                "parsed_steps": [],
                "phases": {},
                "remarks": [
                    {
                        "severity": "error",
                        "message": "Парсинг LaTeX не дал шагов",
                        "phase": "parsing",
                    }
                ],
                "debate_history": [],
            }
        fact_phase = await self._run_phase(
            phase_name="fact_checking",
            proof_steps=proof_steps,
            debate_history=debate_history,
            context=context,
            formulator_fn=self.formulator.formulator_check_facts,
            critic_fn=self.critic.critic_review_facts,
            max_rounds=max_rounds,
        )
        logic_phase = await self._run_phase(
            phase_name="logic_checking",
            proof_steps=proof_steps,
            debate_history=debate_history,
            context=context,
            formulator_fn=self.formulator.formulator_check_logic,
            critic_fn=self.critic.critic_review_logic,
            max_rounds=max_rounds,
        )
        fact_remarks = self._collect_final_remarks(debate_history, "fact_checking")
        logic_remarks = self._collect_final_remarks(debate_history, "logic_checking")
        all_remarks = fact_remarks + logic_remarks
        normalized_steps = self._attach_used_theorems(
            proof_steps, debate_history, all_remarks
        )
        error_count = sum(1 for r in all_remarks if r.get("severity") == "error")
        warning_count = sum(1 for r in all_remarks if r.get("severity") == "warning")
        is_valid = error_count == 0
        fact_ok = fact_phase["consensus_reached"]
        logic_ok = logic_phase["consensus_reached"]
        base_conf = (
            0.95
            if (fact_ok and logic_ok)
            else (0.80 if (fact_ok or logic_ok) else 0.65)
        )
        confidence = round(
            max(0.0, base_conf - error_count * 0.20 - warning_count * 0.04), 2
        )
        if not all_remarks:
            summary = "✅ Доказательство принято без замечаний"
        elif is_valid:
            summary = f"⚠ Принято с предупреждениями: {warning_count} замечан{'ие' if warning_count == 1 else 'ий'}"
        else:
            fact_err = sum(1 for r in fact_remarks if r.get("severity") == "error")
            logic_err = sum(1 for r in logic_remarks if r.get("severity") == "error")
            parts = []
            if fact_err:
                parts.append(
                    f"{fact_err} фактическ{'ая ошибка' if fact_err == 1 else 'их ошибок'}"
                )
            if logic_err:
                parts.append(
                    f"{logic_err} логическ{'ая ошибка' if logic_err == 1 else 'их ошибок'}"
                )
            summary = f"❌ Не принято: {', '.join(parts)}"
        iteration_recommendation = None
        if not is_valid:
            error_steps = sorted(
                {
                    r["step_id"]
                    for r in all_remarks
                    if r.get("severity") == "error" and r.get("step_id") is not None
                }
            )
            global_errors = [
                r
                for r in all_remarks
                if r.get("severity") == "error" and r.get("step_id") is None
            ]
            parts = []
            if error_steps:
                parts.append(f"Исправьте шаги {error_steps}")
            if global_errors:
                parts.append("Пересмотрите общую структуру доказательства")
            iteration_recommendation = ". ".join(parts) or "Требуется доработка"
        logger.info("\n%s", "#" * 60)
        logger.info(
            "# is_valid=%s, confidence=%s, errors=%s, warnings=%s",
            is_valid,
            confidence,
            error_count,
            warning_count,
        )
        logger.info("%s\n", "#" * 60)
        return {
            "proof_id": proof_id,
            "is_valid": is_valid,
            "confidence_score": confidence,
            "summary": summary,
            "iteration_recommendation": iteration_recommendation,
            "parsed_steps": normalized_steps,
            "phases": {
                "fact_checking": {
                    "consensus_reached": fact_phase["consensus_reached"],
                    "rounds_taken": fact_phase["rounds_taken"],
                    "error_count": sum(
                        1 for r in fact_remarks if r.get("severity") == "error"
                    ),
                    "warning_count": sum(
                        1 for r in fact_remarks if r.get("severity") == "warning"
                    ),
                },
                "logic_checking": {
                    "consensus_reached": logic_phase["consensus_reached"],
                    "rounds_taken": logic_phase["rounds_taken"],
                    "error_count": sum(
                        1 for r in logic_remarks if r.get("severity") == "error"
                    ),
                    "warning_count": sum(
                        1 for r in logic_remarks if r.get("severity") == "warning"
                    ),
                },
            },
            "remarks": all_remarks,
            "debate_history": debate_history,
        }
