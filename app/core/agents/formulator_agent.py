import asyncio
import json
import logging
from typing import Any

from app.core.clients.db.rag.vector_store import VectorStore
from app.core.clients.llm.base import BaseLLMClient
from app.core.utils.proof_utils import ProofUtils

logger = logging.getLogger(__name__)


class FormulatorAgent:
    """Agent that formulates and checks facts and logic of proofs."""

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: BaseLLMClient,
        proof_utils: ProofUtils,
    ) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.proof_utils = proof_utils

    @staticmethod
    def _context_header(context: dict[str, Any]) -> str:
        return (
            f"Область: {context.get('topic', 'математика')}. "
            f"Уровень строгости: {context.get('level', 'стандартный')}."
        )

    @staticmethod
    def _step_display(step: dict[str, Any]) -> str:
        latex = step.get("content_latex", "")
        latex_hint = f"\n   LaTeX: {latex}" if latex else ""
        return (
            f"{step.get('content', '')} [тип: {step.get('step_type', '?')}]{latex_hint}"
        )

    @staticmethod
    def _build_proof_summary(proof_steps: list[dict[str, Any]]) -> str:
        lines = []
        for i, s in enumerate(proof_steps):
            lt = s.get("content_latex", "")
            latex_hint = f" [{lt[:100]}{'…' if len(lt) > 100 else ''}]" if lt else ""
            lines.append(
                f"Шаг {i} ({s.get('step_type', '?')}): {s.get('content', '')}{latex_hint} | обоснование: {s.get('justification', '—')}"
            )
        return "\n".join(lines)

    async def _extract_facts_from_step(
        self, step: dict[str, Any], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        latex_block = (
            f"\nLaTeX: {step['content_latex']}" if step.get("content_latex") else ""
        )
        prompt = f"""
{self._context_header(context)}

Проанализируй шаг математического доказательства и выдели все факты,
теоремы, леммы, аксиомы или свойства, на которые явно или неявно опирается этот шаг.

Шаг: {step.get("content", "")}{latex_block}
Тип шага: {step.get("step_type", "assertion")}
Явное обоснование автора: {step.get("justification", "") or "не указано"}

Верни JSON:
{{
    "facts": [
        {{
            "name": "название факта/теоремы (только текст, без LaTeX)",
            "query": "поисковый запрос для базы знаний (суть без LaTeX-команд)",
            "how_used": "как именно используется в этом шаге"
        }}
    ]
}}

Если шаг не опирается ни на какой внешний факт — верни {{"facts": []}}.
"""
        result = await self.llm_client.call(
            prompt,
            system_instruction=(
                f"Ты — математик-аналитик, специалист в «{context.get('topic', 'математика')}». "
                "Выделяй только реально используемые факты, без домыслов. "
                "Отвечай только валидным JSON без LaTeX в текстовых полях."
            ),
        )
        facts = result.get("facts", [])
        return facts if isinstance(facts, list) else []

    async def _verify_fact_with_rag(
        self, fact: dict[str, str], step: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        query = fact.get("query") or fact.get("name", "")
        theorems = await asyncio.to_thread(self.vector_store.search, query, 3)
        if not theorems:
            return {
                "fact_name": fact.get("name"),
                "is_valid": False,
                "severity": "error",
                "message": f"Факт «{fact.get('name')}» не найден в базе знаний",
                "suggestion": "Проверьте название теоремы или укажите явное обоснование",
                "rag_references": [],
            }
        theorems_text = "\n\n".join(
            f"[{i + 1}] {t['name']} ({t.get('type', '?')}):\n{t['statement']}"
            for i, t in enumerate(theorems)
        )
        latex_block = (
            f"\nLaTeX шага: {step['content_latex']}"
            if step.get("content_latex")
            else ""
        )
        prompt = f"""
{self._context_header(context)}

Шаг доказательства: {step.get("content", "")}{latex_block}
Используемый факт: {fact.get("name")}
Как используется: {fact.get("how_used", "")}

Найденные теоремы из базы знаний:
{theorems_text}

Вопросы:
1. Есть ли среди найденных теорем та, что соответствует используемому факту?
2. Корректно ли применён факт с учётом уровня «{context.get("level", "стандартный")}»?
   Выполнены ли все условия теоремы?
3. Если нет — что именно не так?

Верни JSON (все строки — обычный текст, без LaTeX-команд):
{{
    "is_valid": true,
    "matched_theorem": "название совпавшей теоремы или null",
    "severity": "error|warning|info",
    "message": "краткое описание",
    "missing_conditions": ["список невыполненных условий"],
    "suggestion": "как исправить"
}}
"""
        result = await self.llm_client.call(
            prompt,
            system_instruction=(
                f"Ты — строгий верификатор математических доказательств "
                f"в области «{context.get('topic', 'математика')}». "
                "Проверяй точное соответствие условий теорем. "
                "Отвечай только валидным JSON, без LaTeX в текстовых полях."
            ),
        )
        result["fact_name"] = fact.get("name")
        result["rag_references"] = [t.get("name", "") for t in theorems]
        return result

    async def formulator_check_facts(
        self,
        proof_steps: list[dict[str, Any]],
        debate_history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        compressed = self.proof_utils.compress_history(debate_history)
        history_text = self.proof_utils.format_compressed_history(compressed)
        step_results: dict[int, dict] = {}
        logger.info(
            "📋 Формулировщик [факты]: %s шагов, споров=%s",
            len(proof_steps),
            len(compressed),
        )
        for idx, step in enumerate(proof_steps):
            facts = await self._extract_facts_from_step(step, context)
            remarks: list[dict] = []
            facts_found: list[dict] = []
            if not facts:
                step_results[idx] = {"is_valid": True, "facts_found": [], "remarks": []}
                continue
            for fact in facts:
                ver = await self._verify_fact_with_rag(fact, step, context)
                facts_found.append(
                    {
                        "name": fact.get("name"),
                        "matched_theorem": ver.get("matched_theorem"),
                        "rag_references": ver.get("rag_references", []),
                    }
                )
                if not ver.get("is_valid", True):
                    remarks.append(
                        {
                            "step_id": idx,
                            "fact_name": fact.get("name"),
                            "source": "fact_verification",
                            "severity": ver.get("severity", "error"),
                            "message": ver.get("message", ""),
                            "missing_conditions": ver.get("missing_conditions", []),
                            "suggestion": ver.get("suggestion", ""),
                            "rag_references": ver.get("rag_references", []),
                        }
                    )
            active_dispute = next(
                (
                    d
                    for d in compressed
                    if d["phase"] == "fact_checking" and d["step_key"] == str(idx)
                ),
                None,
            )
            if active_dispute and active_dispute.get("critic_remarks") and remarks:
                critic_text = "\n".join(
                    f"- [{r.get('severity', '?')}] {r.get('message', '')}"
                    for r in active_dispute["critic_remarks"]
                )
                reconsider_prompt = f"""
{self._context_header(context)}

{history_text}

Ты — формулировщик. Критик не согласился с твоими замечаниями по шагу {idx}.

Шаг: {self._step_display(step)}

Возражения критика:
{critic_text}

Твои текущие замечания:
{json.dumps(remarks, ensure_ascii=False, indent=2)}

Пересмотри замечания. Если критик прав — скорректируй или сними.
Если не прав — сохрани и усиль аргументацию.

Верни JSON (все строки — обычный текст, без LaTeX-команд):
{{
    "revised_remarks": [
        {{
            "step_id": {idx},
            "fact_name": "название факта",
            "source": "fact_verification",
            "severity": "error|warning|info",
            "message": "обновлённое описание",
            "missing_conditions": [],
            "suggestion": "обновлённое предложение",
            "rag_references": []
        }}
    ]
}}
"""
                revision = await self.llm_client.call(
                    reconsider_prompt,
                    system_instruction=(
                        f"Ты — математик-формулировщик в «{context.get('topic', 'математика')}». "
                        "Будь открыт к аргументам, но стой на своём если уверен. "
                        "Отвечай только валидным JSON, без LaTeX в текстовых полях."
                    ),
                )
                revised = revision.get("revised_remarks")
                if isinstance(revised, list):
                    remarks = revised
                    logger.debug("Шаг %s: формулировщик пересмотрел замечания", idx)
            step_results[idx] = {
                "is_valid": not any(r.get("severity") == "error" for r in remarks),
                "facts_found": facts_found,
                "remarks": remarks,
            }
        overall_valid = all(v["is_valid"] for v in step_results.values())
        logger.info("Формулировщик [факты]: overall_valid=%s", overall_valid)
        return {
            "step_results": step_results,
            "phase": "fact_checking",
            "agent": "formulator",
            "overall_verdict": overall_valid,
        }

    async def formulator_check_logic(
        self,
        proof_steps: list[dict[str, Any]],
        debate_history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        compressed = self.proof_utils.compress_history(debate_history)
        history_text = self.proof_utils.format_compressed_history(compressed)
        proof_summary = self._build_proof_summary(proof_steps)
        logger.info("🔗 Формулировщик [логика]: %s шагов", len(proof_steps))
        global_prompt = f"""
{self._context_header(context)}

{history_text}

Краткое резюме доказательства:
{proof_summary}

Проверь логическую структуру с учётом уровня «{context.get("level", "стандартный")}»:
1. Логически ли следует каждый шаг из предыдущих?
2. Есть ли необоснованные прыжки в рассуждениях?
3. Достигается ли заявленный результат?
4. Нет ли порочного круга?
5. Соответствует ли строгость уровню «{context.get("level", "стандартный")}»?

Верни JSON (все строки — обычный текст, без LaTeX):
{{
    "global_issues": [
        {{
            "severity": "error|warning|info",
            "message": "описание проблемы",
            "affected_steps": [0, 1],
            "suggestion": "как исправить"
        }}
    ],
    "step_transitions": [
        {{
            "from_step": 0,
            "to_step": 1,
            "is_valid": true,
            "message": "комментарий"
        }}
    ]
}}
"""
        global_analysis = await self.llm_client.call(
            global_prompt,
            system_instruction=(
                f"Ты — математик-логик, специалист в «{context.get('topic', 'математика')}». "
                "Анализируй структуру строго и систематично. "
                "Отвечай только валидным JSON, без LaTeX в текстовых полях."
            ),
        )
        step_results: dict[int, dict] = {}
        for idx, step in enumerate(proof_steps):
            window = self.proof_utils.get_window(
                proof_steps, idx, window_size=ProofUtils.SLIDING_WINDOW_SIZE
            )
            window_text = self.proof_utils.format_window(window, proof_steps)
            active_dispute = next(
                (
                    d
                    for d in compressed
                    if d["phase"] == "logic_checking" and d["step_key"] == str(idx)
                ),
                None,
            )
            critic_context = ""
            if active_dispute and active_dispute.get("critic_remarks"):
                critic_context = (
                    "Возражения критика из предыдущего раунда:\n"
                    + "\n".join(
                        f"  [{r.get('severity', '?')}] {r.get('message', '')}"
                        for r in active_dispute["critic_remarks"]
                    )
                )
            step_prompt = f"""
{self._context_header(context)}

{history_text}

{critic_context}

Предыдущие шаги (окно {ProofUtils.SLIDING_WINDOW_SIZE}):
{window_text}

Текущий шаг {idx}:
{self._step_display(step)}

Проверь с учётом уровня «{context.get("level", "стандартный")}»:
1. Логически ли следует этот шаг из предыдущих?
2. Достаточно ли обоснование («{step.get("justification", "не указано")}»)?
3. Нет ли неявных допущений, неприемлемых на данном уровне?

Верни JSON (все строки — обычный текст, без LaTeX):
{{
    "is_valid": true,
    "severity": "error|warning|info",
    "issues": [
        {{
            "type": "logic_gap|missing_step|circular_reasoning|unjustified_assumption|none",
            "message": "описание",
            "suggestion": "как исправить"
        }}
    ]
}}
"""
            step_analysis = await self.llm_client.call(
                step_prompt,
                system_instruction=(
                    f"Ты — строгий математический критик в «{context.get('topic', 'математика')}». "
                    "Проверяй логику без обращения к базе теорем. "
                    "Отвечай только валидным JSON, без LaTeX в текстовых полях."
                ),
            )
            issues = (
                step_analysis.get("issues", [])
                if isinstance(step_analysis.get("issues"), list)
                else []
            )
            remarks = [
                {
                    "step_id": idx,
                    "source": "logic_analysis",
                    "severity": issue.get(
                        "severity", step_analysis.get("severity", "warning")
                    ),
                    "type": issue.get("type", "logic_gap"),
                    "message": issue.get("message", ""),
                    "suggestion": issue.get("suggestion", ""),
                }
                for issue in issues
                if issue.get("type") != "none"
            ]
            step_results[idx] = {
                "is_valid": step_analysis.get("is_valid", True)
                and not any(r.get("severity") == "error" for r in remarks),
                "remarks": remarks,
            }
        global_remarks = []
        for issue in (
            global_analysis.get("global_issues", [])
            if isinstance(global_analysis.get("global_issues"), list)
            else []
        ):
            if issue.get("severity") in ("error", "warning"):
                global_remarks.append(
                    {
                        "source": "global_logic_analysis",
                        "severity": issue["severity"],
                        "message": issue.get("message", ""),
                        "affected_steps": issue.get("affected_steps", []),
                        "suggestion": issue.get("suggestion", ""),
                    }
                )
                if issue["severity"] == "error":
                    for affected_idx in issue.get("affected_steps", []):
                        if affected_idx in step_results:
                            step_results[affected_idx]["is_valid"] = False
        overall_valid = all(v["is_valid"] for v in step_results.values()) and not any(
            r["severity"] == "error" for r in global_remarks
        )
        logger.info(
            "Формулировщик [логика]: overall_valid=%s, глоб.=%s",
            overall_valid,
            len(global_remarks),
        )
        return {
            "step_results": step_results,
            "global_remarks": global_remarks,
            "step_transitions": global_analysis.get("step_transitions", []),
            "phase": "logic_checking",
            "agent": "formulator",
            "overall_verdict": overall_valid,
        }
