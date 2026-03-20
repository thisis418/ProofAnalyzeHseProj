import asyncio
import json
import logging
from typing import Any

from app.core.clients.db.rag.vector_store import VectorStore
from app.core.clients.llm.base import BaseLLMClient
from app.core.utils.proof_utils import ProofUtils

logger = logging.getLogger(__name__)


class CriticAgent:
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

    @staticmethod
    def _merge_retrieval_results(
        batches: list[list[dict[str, Any]]], top_k: int
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for batch in batches:
            for item in batch:
                key = f"{item.get('type','')}|{item.get('name','')}"
                prev = merged.get(key)
                if prev is None or float(item.get("score", 0.0)) > float(
                    prev.get("score", 0.0)
                ):
                    merged[key] = item
        ordered = sorted(
            merged.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True
        )
        return ordered[:top_k]

    async def _search_rag_multilingual(
        self, query: str, top_k: int, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        queries = await self.proof_utils.build_rag_queries(query, context)
        if not queries:
            return []
        batches: list[list[dict[str, Any]]] = []
        for q in queries:
            batches.append(await asyncio.to_thread(self.vector_store.search, q, top_k))
        return self._merge_retrieval_results(batches, top_k=top_k)

    async def _critic_verify_fact_independently(
        self,
        fact_name: str,
        step: dict[str, Any],
        formulator_remark: dict[str, Any] | None,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        theorems = await self._search_rag_multilingual(
            fact_name, top_k=3, context=context
        )
        theorems_text = (
            "\n\n".join(
                f"[{i + 1}] {t['name']} ({t.get('type', '?')}):\n{t['statement']}"
                for i, t in enumerate(theorems)
            )
            if theorems
            else "Теоремы не найдены."
        )
        formulator_text = (
            json.dumps(formulator_remark, ensure_ascii=False, indent=2)
            if formulator_remark
            else "Формулировщик не нашёл проблем с этим фактом."
        )
        latex_block = (
            f"\nLaTeX шага: {step['content_latex']}"
            if step.get("content_latex")
            else ""
        )
        prompt = f"""
{self._context_header(context)}

Шаг доказательства: {step.get("content", "")}{latex_block}
Обоснование автора: {step.get("justification", "не указано")}

Анализируемый факт: {fact_name}

Замечание формулировщика:
{formulator_text}

Результаты независимого поиска в базе знаний:
{theorems_text}

Твоя задача:
1. Независимо оцени корректность использования факта.
2. Согласен ли ты с замечанием формулировщика?
3. Если формулировщик не нашёл проблем — нет ли всё же ошибки?

Верни JSON:
{{
    "agrees_with_formulator": true,
    "independent_verdict": "ok|error|warning",
    "critic_remarks": [
        {{
            "severity": "error|warning|info",
            "message": "твоё замечание",
            "rag_references": ["теоремы, на которые опираешься"]
        }}
    ],
    "reasoning": "краткое объяснение позиции"
}}
"""
        result = await self.llm_client.call(
            prompt,
            system_instruction=(
                f"Ты — независимый критик математических доказательств в области «{context.get('topic', 'математика')}». "
                "Проверяй факты сам, не соглашайся автоматически. "
                "Отвечай только валидным JSON, без LaTeX в текстовых полях."
            ),
        )
        result["fact_name"] = fact_name
        result["rag_references_found"] = [t.get("name", "") for t in theorems]
        return result

    async def critic_review_facts(
        self,
        proof_steps: list[dict[str, Any]],
        formulator_result: dict[str, Any],
        debate_history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        compressed = self.proof_utils.compress_history(debate_history)
        history_text = self.proof_utils.format_compressed_history(compressed)
        step_reviews: dict[str, dict] = {}
        logger.info(
            "🔎 Критик [факты]: %s шагов, споров=%s", len(proof_steps), len(compressed)
        )
        for idx, step in enumerate(proof_steps):
            formulator_step = formulator_result.get("step_results", {}).get(idx, {})
            formulator_remarks = formulator_step.get("remarks", [])
            facts_found = formulator_step.get("facts_found", [])
            fact_names = {
                r.get("fact_name") for r in formulator_remarks if r.get("fact_name")
            } | {f.get("name") for f in facts_found if f.get("name")}
            if not fact_names:
                latex_block = (
                    f"\nLaTeX: {step['content_latex']}"
                    if step.get("content_latex")
                    else ""
                )
                check_prompt = f"""
{self._context_header(context)}

Шаг: {step.get("content", "")}{latex_block}
Обоснование: {step.get("justification", "не указано")}

Формулировщик решил, что этот шаг не опирается ни на какую теорему.
Согласен ли ты? Если нет — назови пропущенные факты.

Верни JSON:
{{
    "agrees_no_facts": true,
    "missed_facts": ["названия пропущенных фактов"]
}}
"""
                check = await self.llm_client.call(
                    check_prompt,
                    system_instruction=f"Ты — внимательный математик-критик в «{context.get('topic', 'математика')}».",
                )
                if not check.get("agrees_no_facts", True):
                    fact_names = set(check.get("missed_facts", []))
            if not fact_names:
                step_reviews[str(idx)] = {
                    "agrees_with_formulator": True,
                    "critic_remarks": [],
                    "rag_references_found": [],
                }
                continue
            per_fact = []
            for fact_name in fact_names:
                matching = next(
                    (r for r in formulator_remarks if r.get("fact_name") == fact_name),
                    None,
                )
                per_fact.append(
                    await self._critic_verify_fact_independently(
                        fact_name, step, matching, context
                    )
                )
            all_agree = all(r.get("agrees_with_formulator", True) for r in per_fact)
            all_remarks: list[dict] = []
            all_refs: list[str] = []
            for fr in per_fact:
                all_remarks.extend(fr.get("critic_remarks", []))
                all_refs.extend(fr.get("rag_references_found", []))
            active_dispute = next(
                (
                    d
                    for d in compressed
                    if d["phase"] == "fact_checking" and d["step_key"] == str(idx)
                ),
                None,
            )
            if active_dispute and not all_agree:
                reconcile_prompt = f"""
{self._context_header(context)}

{history_text}

По шагу {idx} продолжается спор.

Шаг: {self._step_display(step)}

Твои текущие возражения:
{json.dumps(all_remarks, ensure_ascii=False, indent=2)}

Последние замечания формулировщика:
{json.dumps(active_dispute.get("formulator_remarks", []), ensure_ascii=False, indent=2)}

Стоит ли тебе скорректировать позицию в ответ?

Верни JSON:
{{
    "final_agrees": true,
    "final_remarks": [
        {{
            "severity": "error|warning|info",
            "message": "итоговое замечание",
            "rag_references": []
        }}
    ],
    "reasoning": "почему изменил или не изменил позицию"
}}
"""
                reconcile = await self.llm_client.call(
                    reconcile_prompt,
                    system_instruction=(
                        f"Ты — критик в «{context.get('topic', 'математика')}». "
                        "Будь принципиальным, но справедливым. "
                        "Отвечай только валидным JSON, без LaTeX в текстовых полях."
                    ),
                )
                if "final_agrees" in reconcile:
                    all_agree = reconcile["final_agrees"]
                    all_remarks = reconcile.get("final_remarks", all_remarks)
            step_reviews[str(idx)] = {
                "agrees_with_formulator": all_agree,
                "critic_remarks": all_remarks,
                "rag_references_found": list(set(all_refs)),
            }
        consensus_reached = all(
            r.get("agrees_with_formulator", True) for r in step_reviews.values()
        )
        logger.info("Критик [факты]: consensus=%s", consensus_reached)
        return {
            "step_reviews": step_reviews,
            "phase": "fact_checking",
            "agent": "critic",
            "consensus_reached": consensus_reached,
        }

    async def critic_review_logic(
        self,
        proof_steps: list[dict[str, Any]],
        formulator_result: dict[str, Any],
        debate_history: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        compressed = self.proof_utils.compress_history(debate_history)
        history_text = self.proof_utils.format_compressed_history(compressed)
        proof_summary = self._build_proof_summary(proof_steps)
        logger.info("🔗 Критик [логика]: %s шагов", len(proof_steps))
        global_formulator_remarks = formulator_result.get("global_remarks", [])
        global_prompt = f"""
{self._context_header(context)}

{history_text}

Краткое резюме доказательства:
{proof_summary}

Замечания формулировщика по глобальной логике:
{json.dumps(global_formulator_remarks, ensure_ascii=False, indent=2)}

Твоя задача:
1. Независимо оцени логическую структуру всего доказательства.
2. Согласен ли ты с глобальными замечаниями формулировщика?
3. Есть ли проблемы, которые формулировщик пропустил?

Верни JSON:
{{
    "agrees_with_formulator_global": true,
    "critic_global_remarks": [
        {{
            "severity": "error|warning|info",
            "message": "замечание",
            "affected_steps": [0, 1]
        }}
    ],
    "reasoning": "краткое объяснение"
}}
"""
        global_review = await self.llm_client.call(
            global_prompt,
            system_instruction=(
                f"Ты — независимый логик, специалист в «{context.get('topic', 'математика')}». "
                "Оценивай структуру строго и честно. "
                "Отвечай только валидным JSON, без LaTeX в текстовых полях."
            ),
        )
        step_reviews: dict[str, dict] = {}
        for idx, step in enumerate(proof_steps):
            window = self.proof_utils.get_window(
                proof_steps, idx, window_size=ProofUtils.SLIDING_WINDOW_SIZE
            )
            window_text = self.proof_utils.format_window(window, proof_steps)
            formulator_step = formulator_result.get("step_results", {}).get(idx, {})
            formulator_remarks = formulator_step.get("remarks", [])
            active_dispute = next(
                (
                    d
                    for d in compressed
                    if d["phase"] == "logic_checking" and d["step_key"] == str(idx)
                ),
                None,
            )
            history_context = (
                f"Текущий статус спора по шагу {idx}: {active_dispute['status']}."
                if active_dispute
                else ""
            )
            step_prompt = f"""
{self._context_header(context)}

{history_text}

{history_context}

Предыдущие шаги (окно {ProofUtils.SLIDING_WINDOW_SIZE}):
{window_text}

Текущий шаг {idx}:
{self._step_display(step)}

Замечания формулировщика:
{json.dumps(formulator_remarks, ensure_ascii=False, indent=2) if formulator_remarks else "Нет замечаний"}

Твоя задача:
1. Независимо проверь логику перехода к этому шагу.
2. Согласен ли ты с замечаниями формулировщика?
3. Есть ли проблемы, которые формулировщик не заметил?

Верни JSON:
{{
    "agrees_with_formulator": true,
    "independent_is_valid": true,
    "critic_remarks": [
        {{
            "severity": "error|warning|info",
            "type": "logic_gap|missing_step|circular_reasoning|unjustified_assumption|formulator_error",
            "message": "замечание",
            "suggestion": "как исправить"
        }}
    ],
    "reasoning": "краткое объяснение"
}}
"""
            step_review = await self.llm_client.call(
                step_prompt,
                system_instruction=(
                    f"Ты — строгий математический критик в «{context.get('topic', 'математика')}». "
                    "Не используй базу теорем — только логику. "
                    "Отвечай только валидным JSON, без LaTeX в текстовых полях."
                ),
            )
            step_reviews[str(idx)] = {
                "agrees_with_formulator": step_review.get(
                    "agrees_with_formulator", True
                ),
                "independent_is_valid": step_review.get("independent_is_valid", True),
                "critic_remarks": step_review.get("critic_remarks", []),
            }
        steps_consensus = all(
            r.get("agrees_with_formulator", True) for r in step_reviews.values()
        )
        global_consensus = global_review.get("agrees_with_formulator_global", True)
        consensus_reached = steps_consensus and global_consensus
        logger.info("Критик [логика]: consensus=%s", consensus_reached)
        return {
            "step_reviews": step_reviews,
            "global_review": {
                "agrees": global_review.get("agrees_with_formulator_global", True),
                "critic_global_remarks": global_review.get("critic_global_remarks", []),
            },
            "phase": "logic_checking",
            "agent": "critic",
            "consensus_reached": consensus_reached,
        }
