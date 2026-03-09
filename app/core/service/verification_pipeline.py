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
            "parsed_steps": proof_steps,
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
