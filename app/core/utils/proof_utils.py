import logging
import re
from typing import Any

from app.core.clients.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)


class ProofUtils:
    """Shared proof parsing and debate-history helpers."""

    SLIDING_WINDOW_SIZE: int = 5

    _BLOCK_ENVS = re.compile(
        r"\\begin\{(align\*?|equation\*?|gather\*?|multline\*?|cases|array|matrix|pmatrix|bmatrix)\}"
        r"(.*?)"
        r"\\end\{\1\}",
        re.DOTALL,
    )
    _SENTENCE_SPLIT_RE = re.compile(
        r"(?<=[.!?])\s+"
        r"|(?<=\.)\s*\n\s*"
        r"|\\\\\s*\n"
        r"|\n{2,}"
    )
    _STRIP_COMMANDS = re.compile(
        r"\\(?:label|ref|eqref|tag|nonumber|hline|vspace|hspace|noindent|medskip|bigskip|smallskip)"
        r"(?:\{[^}]*\})?"
    )
    _PROOF_ENV = re.compile(r"\\begin\{proof\}(.*?)\\end\{proof\}", re.DOTALL)

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm_client = llm_client

    @classmethod
    def _extract_proof_body(cls, latex: str) -> str:
        match = cls._PROOF_ENV.search(latex)
        return match.group(1).strip() if match else latex.strip()

    @classmethod
    def _protect_math_blocks(cls, text: str) -> tuple[str, dict[str, str]]:
        placeholders: dict[str, str] = {}
        counter = [0]

        def replace(match: re.Match) -> str:
            key = f"MATHBLOCK_{counter[0]}"
            placeholders[key] = match.group(0)
            counter[0] += 1
            return key

        return cls._BLOCK_ENVS.sub(replace, text), placeholders

    @staticmethod
    def _restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
        for key, value in placeholders.items():
            text = text.replace(key, value)
        return text

    @classmethod
    def _clean_latex_sentence(cls, sentence: str) -> str:
        sentence = cls._STRIP_COMMANDS.sub("", sentence)
        sentence = re.sub(r"\s+", " ", sentence).strip()
        if sentence.count("$") % 2 != 0:
            sentence = sentence.rstrip("$").strip()
        return sentence

    def latex_to_sentences(self, latex: str) -> list[str]:
        body = self._extract_proof_body(latex)
        protected, placeholders = self._protect_math_blocks(body)
        parts = self._SENTENCE_SPLIT_RE.split(protected)
        sentences: list[str] = []
        for part in parts:
            restored = self._restore_placeholders(part, placeholders)
            cleaned = self._clean_latex_sentence(restored)
            if cleaned:
                sentences.append(cleaned)
        return sentences

    def _fallback_steps(self, sentences: list[str]) -> list[dict[str, Any]]:
        return [
            {
                "content": sentence,
                "content_latex": sentence,
                "justification": "",
                "step_type": "assertion",
                "source_indices": [index],
                "line_number": index,
            }
            for index, sentence in enumerate(sentences)
        ]

    async def sentences_to_steps(
        self, sentences: list[str], full_latex: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        topic = context.get("topic", "математика")
        level = context.get("level", "стандартный")
        numbered = "\n".join(
            f"[{i}] {sentence}" for i, sentence in enumerate(sentences)
        )
        prompt = f"""
Тебе дано математическое доказательство в LaTeX и список его предложений с индексами.

Область математики: {topic}
Уровень строгости: {level}

=== Полный LaTeX доказательства ===
{full_latex}

=== Пронумерованные предложения ===
{numbered}

Задача: разбей доказательство на логические шаги.
Правила:
- Несколько предложений можно объединять в один шаг, если это одна мысль.
- Каждый шаг должен быть атомарным.
- В поле content_latex сохрани оригинальный LaTeX.
- В поле content пиши только обычный текст без LaTeX-команд.
- Все обратные слеши в JSON-строках обязательно экранируй по правилам JSON.
  Пример корректного JSON: "\\\\langle g \\\\rangle".

Формат ответа:
{{
  "steps": [
    {{
      "content": "plain-text описание шага",
      "content_latex": "LaTeX-фрагмент",
      "justification": "теорема/лемма/аксиома или пусто",
      "step_type": "definition|assumption|assertion|implication|conclusion",
      "source_indices": [0, 1]
    }}
  ]
}}
"""
        result = await self.llm_client.call(
            prompt,
            system_instruction=(
                f"Ты — математик, специалист в области «{topic}». "
                f"Разбивай доказательства на чёткие логические шаги с учётом уровня строгости «{level}»."
            ),
        )
        if "error" in result:
            logger.error("LLM step parsing failed: %s", result["error"])
            return self._fallback_steps(sentences)
        raw_steps = result.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            return self._fallback_steps(sentences)
        steps: list[dict[str, Any]] = []
        for index, raw_step in enumerate(raw_steps):
            source_indices = (
                raw_step.get("source_indices", []) if isinstance(raw_step, dict) else []
            )
            if not isinstance(source_indices, list):
                source_indices = []
            steps.append(
                {
                    "content": raw_step.get("content", "")
                    if isinstance(raw_step, dict)
                    else "",
                    "content_latex": raw_step.get("content_latex", "")
                    if isinstance(raw_step, dict)
                    else "",
                    "justification": raw_step.get("justification", "")
                    if isinstance(raw_step, dict)
                    else "",
                    "step_type": raw_step.get("step_type", "assertion")
                    if isinstance(raw_step, dict)
                    else "assertion",
                    "source_indices": source_indices,
                    "line_number": source_indices[0] if source_indices else index,
                }
            )
        return steps or self._fallback_steps(sentences)

    async def parse_latex_proof(
        self, latex: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        sentences = self.latex_to_sentences(latex)
        logger.info("parse_latex_proof: %s sentences", len(sentences))
        steps = await self.sentences_to_steps(sentences, latex, context)
        logger.info("parse_latex_proof: %s steps", len(steps))
        return steps

    @classmethod
    def get_window(
        cls,
        proof_steps: list[dict[str, Any]],
        current_idx: int,
        window_size: int | None = None,
    ) -> list[dict[str, Any]]:
        window_size = window_size or cls.SLIDING_WINDOW_SIZE
        start = max(0, current_idx - window_size)
        window = list(proof_steps[start:current_idx])
        if start > 0 and proof_steps:
            window = [proof_steps[0]] + window
        return window

    @staticmethod
    def format_window(
        window: list[dict[str, Any]], proof_steps: list[dict[str, Any]]
    ) -> str:
        if not window:
            return "Нет предыдущих шагов."
        lines: list[str] = []
        for step in window:
            try:
                real_idx = proof_steps.index(step)
            except ValueError:
                real_idx = "?"
            latex_hint = ""
            if step.get("content_latex"):
                latex = step["content_latex"]
                latex_hint = (
                    f"\n   LaTeX: {latex[:120]}{'…' if len(latex) > 120 else ''}"
                )
            lines.append(
                f"Шаг {real_idx} ({step.get('step_type', '?')}): {step.get('content', '')}{latex_hint}"
            )
        return "\n".join(lines)

    @staticmethod
    def compress_history(debate_history: list[dict]) -> list[dict]:
        if not debate_history:
            return []
        disputes: dict[tuple[str, str], dict[str, list]] = {}
        for entry in debate_history:
            phase = entry.get("phase", "?")
            agent = entry.get("agent")
            round_num = entry.get("round", 0)
            if agent == "formulator":
                for step_id, step_data in entry.get("step_results", {}).items():
                    key = (phase, str(step_id))
                    disputes.setdefault(key, {"formulator": [], "critic": []})
                    if step_data.get("remarks"):
                        disputes[key]["formulator"].append(
                            {
                                "round": round_num,
                                "remarks": step_data["remarks"],
                                "is_valid": step_data.get("is_valid", True),
                            }
                        )
                if entry.get("global_remarks"):
                    key = (phase, "global")
                    disputes.setdefault(key, {"formulator": [], "critic": []})
                    disputes[key]["formulator"].append(
                        {"round": round_num, "remarks": entry["global_remarks"]}
                    )
            elif agent == "critic":
                for step_id, step_data in entry.get("step_reviews", {}).items():
                    key = (phase, str(step_id))
                    disputes.setdefault(key, {"formulator": [], "critic": []})
                    disputes[key]["critic"].append(
                        {
                            "round": round_num,
                            "agrees": step_data.get("agrees_with_formulator", True),
                            "remarks": step_data.get("critic_remarks", []),
                        }
                    )
                global_review = entry.get("global_review", {})
                if global_review and not global_review.get("agrees", True):
                    key = (phase, "global")
                    disputes.setdefault(key, {"formulator": [], "critic": []})
                    disputes[key]["critic"].append(
                        {
                            "round": round_num,
                            "agrees": False,
                            "remarks": global_review.get("critic_global_remarks", []),
                        }
                    )
        compressed: list[dict] = []
        for (phase, key), sides in disputes.items():
            if not sides["critic"]:
                last_formulator = (
                    sides["formulator"][-1] if sides["formulator"] else None
                )
                if last_formulator and last_formulator.get("remarks"):
                    compressed.append(
                        {
                            "phase": phase,
                            "step_key": key,
                            "status": "pending_critic",
                            "formulator_remarks": last_formulator["remarks"],
                            "critic_remarks": [],
                            "round": last_formulator.get("round", 0),
                        }
                    )
                continue
            last_critic = sides["critic"][-1]
            if last_critic.get("agrees", True):
                continue
            last_formulator = (
                sides["formulator"][-1]
                if sides["formulator"]
                else {"remarks": [], "round": 0}
            )
            compressed.append(
                {
                    "phase": phase,
                    "step_key": key,
                    "status": "disputed",
                    "formulator_remarks": last_formulator.get("remarks", []),
                    "critic_remarks": last_critic.get("remarks", []),
                    "round": max(
                        last_formulator.get("round", 0), last_critic.get("round", 0)
                    ),
                }
            )
        compressed.sort(
            key=lambda item: (item["round"], item["phase"], str(item["step_key"]))
        )
        return compressed

    @staticmethod
    def format_compressed_history(compressed: list[dict]) -> str:
        if not compressed:
            return "Активных разногласий нет (первая итерация или консенсус достигнут)."
        lines = ["=== Текущие разногласия (сжатая история спора) ==="]
        for item in compressed:
            phase_label = "Факты" if item["phase"] == "fact_checking" else "Логика"
            key = item["step_key"]
            status = "⚡ спор" if item["status"] == "disputed" else "⏳ ожидает"
            step_label = f"Шаг {key}" if key != "global" else "Глобально"
            lines.append(
                f"\n[{phase_label} | {step_label} | раунд {item['round']} | {status}]"
            )
            if item["formulator_remarks"]:
                lines.append("  Формулировщик:")
                for remark in item["formulator_remarks"]:
                    lines.append(
                        f"    [{remark.get('severity', '?')}] {remark.get('message', '')}"
                    )
                    if remark.get("suggestion"):
                        lines.append(f"    → {remark['suggestion']}")
            if item["critic_remarks"]:
                lines.append("  Критик:")
                for remark in item["critic_remarks"]:
                    lines.append(
                        f"    [{remark.get('severity', remark.get('type', '?'))}] {remark.get('message', '')}"
                    )
        return "\n".join(lines)
