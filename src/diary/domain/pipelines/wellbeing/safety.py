"""Deterministic wellbeing safety classifier and escalation content (Task 15).

This module is the **rule layer** of the wellbeing feature: it decides the
safety level of a diary week from the normalized (layer 2) text by fixed,
code-defined rules — no LLM is involved anywhere in this module (ADR-006,
``docs/privacy-and-safety.md`` §6-7, ``skills/wellbeing-safety``).

Levels
======

* ``urgent``        — the text contains an explicit statement of self-harm or
                      suicidal intent. The user must receive **fixed**
                      escalation guidance (constants below) whose wording never
                      comes from an LLM;
* ``needs_support`` — the text reports significant distress (sleep problems,
                      anxiety, low mood, exhaustion, isolation, ...) that may
                      warrant supportive attention;
* ``normal``        — none of the above matched.

Conservativeness (ADR-006): missing an ``urgent`` case is unacceptable, a
false positive is acceptable. The classifier therefore uses simple substring
matching on the whole lower-cased document: any occurrence of an urgent
indicator raises the level to ``urgent`` even when the surrounding phrasing is
ambiguous. The lists are deliberately curated to target explicit crisis
phrasings and are extended only together with their tests (synthetic data
only).

Escalation content
==================

``ESCALATION_*`` constants are the fixed, deterministic user-facing crisis
guidance (shown in the urgent report and on the wellbeing page). The
application does not provide medical/psychological care and does not run a
crisis hotline (privacy-and-safety.md §6); the wording only directs the user
to professional help and deliberately names no region-specific phone numbers.

Privacy: the functions in this module accept diary text in memory and never
log it; every constant is static.
"""

from __future__ import annotations

from enum import Enum

#: Fixed title of the deterministic escalation block.
ESCALATION_TITLE = "Важно: в записях есть признаки кризисной ситуации"

#: Fixed, deterministic escalation guidance. Shown verbatim (never LLM text)
#: when the level is ``urgent``. General by design: no region-specific phone
#: numbers and no medical instructions (privacy-and-safety.md §6-7).
ESCALATION_LINES: tuple[str, ...] = (
    "Записи за эту неделю содержат высказывания о желании причинить себе "
    "вред или о суициде. Это приложение не оказывает экстренную помощь и не "
    "заменяет специалиста.",
    "Если вы прямо сейчас находитесь в кризисной ситуации или думаете о "
    "причинении себе вреда, пожалуйста, немедленно обратитесь за помощью:",
    "в службу экстренной помощи по месту жительства (единый номер экстренных служб вашей страны);",
    "к врачу, психологу или психотерапевту;",
    "в ближайший кризисный центр или службу психологической поддержки.",
    "По возможности не оставайтесь в этот момент в одиночестве и обратитесь "
    "к близкому человеку, которому доверяете.",
)

#: Fixed note appended when the level is ``needs_support``: deterministic,
#: supportive, non-alarming, and free of any diagnostic claim.
SUPPORT_NOTE = (
    "В записях недели заметны признаки напряжения и трудного самочувствия. "
    "Отнеситесь к рефлексии ниже как к поддержке, а не как к заключению; "
    "если напряжение сохраняется — это повод обратиться к специалисту."
)

#: Mandatory disclaimer rendered in EVERY wellbeing report (Task 15 required
#: user-facing limitation; ADR-006). Substance fixed by the task list.
DISCLAIMER = (
    "Рефлексия основана только на записях дневника, доступных за этот период. "
    "Это не медицинский и не психологический диагноз и не замена "
    "квалифицированной профессиональной помощи."
)

#: Fixed label of the safety-level line inside a rendered report header.
LEVEL_LABEL = "Уровень"


class SafetyLevel(str, Enum):
    """Deterministic wellbeing safety level (ADR-006; ``urgent`` is the
    strongest level and wins over the other two)."""

    normal = "normal"
    needs_support = "needs_support"
    urgent = "urgent"

    @property
    def display_label(self) -> str:
        """Short Russian caption of the level for the UI (not diagnostic)."""
        return {
            SafetyLevel.normal: "Обычный",
            SafetyLevel.needs_support: "Возможна потребность в поддержке",
            SafetyLevel.urgent: "Требуется внимание",
        }[self]


#: Explicit self-harm / suicidality phrasings → ``urgent``. Substring match on
#: the lower-cased document; the list is intentionally conservative and covers
#: common diary phrasings (including morphological variants of the verbs).
_URGENT_INDICATORS: tuple[str, ...] = (
    "хочу умереть",
    "хочу умирать",
    "не хочу жить",
    "не хочу больше жить",
    "надоело жить",
    "лучше бы я умер",
    "лучше бы не жить",
    "покончить с собой",
    "покончу с собой",
    "покончил с собой",
    "покончила с собой",
    "покончить жизнь самоубийством",
    "свести счёты с жизнью",
    "свести счеты с жизнью",
    "убить себя",
    "убью себя",
    "убил бы себя",
    "причинить себе вред",
    "причиню себе вред",
    "причинил себе вред",
    "причинила себе вред",
    "причинять себе вред",
    "навредить себе",
    "наврежу себе",
    "наношу себе вред",
    "порезать себя",
    "порежу себя",
    "порезала себя",
    "порезал себя",
    "режу себя",
    "самоповреждени",
    "самоубийств",
    "суицид",
    "суицидальн",
    "хочу исчезнуть навсегда",
)

#: Significant-distress phrasings → ``needs_support`` when no urgent
#: indicator matched. Substring match on the lower-cased document.
_NEEDS_SUPPORT_INDICATORS: tuple[str, ...] = (
    "подавлен",
    "выгорел",
    "выгорани",
    "нет сил",
    "постоянная тревог",
    "сильная тревог",
    "тревожн",
    "панические атак",
    "паническая атака",
    "бессонниц",
    "не сплю",
    "плохо сплю",
    "не могу уснуть",
    "не могу заснуть",
    "депресси",
    "всё надоело",
    "все надоело",
    "всё достало",
    "все достало",
    "ненавижу себя",
    "ничего не хочется",
    "не хочется ничего",
    "хочется исчезнуть",
    "хочется убежать",
    "не вижу смысла",
    "внутри пусто",
    "пустота внутри",
    "чувствую себя пуст",
    "сильная усталость",
    "полное истощение",
    "эмоциональное истощение",
    "не справляюсь",
    "срываюсь на",
    "постоянно плач",
    "плакал",
    "плакала",
    "одиночеств",
    "одиноко",
    "никому не нужен",
    "никому не нужна",
    "тяжело на душе",
    "всё валится из рук",
    "ком в горле",
    "не хочу просыпаться",
)


def _contains_any(text: str, indicators: tuple[str, ...]) -> bool:
    """Whether ``text`` contains any of ``indicators`` (substring match)."""
    return any(indicator in text for indicator in indicators)


def classify(text: str) -> SafetyLevel:
    """Deterministic safety level of a normalized diary text.

    The whole document is lower-cased once and scanned with substring rules.
    ``urgent`` wins over everything; ``needs_support`` wins over ``normal``.
    Blank/whitespace-only text is ``normal`` (the caller rejects documents
    without entries before this matters). Same input always returns the same
    level — the classifier never calls an LLM and never depends on provider
    configuration.
    """
    haystack = (text or "").lower()
    if not haystack.strip():
        return SafetyLevel.normal
    if _contains_any(haystack, _URGENT_INDICATORS):
        return SafetyLevel.urgent
    if _contains_any(haystack, _NEEDS_SUPPORT_INDICATORS):
        return SafetyLevel.needs_support
    return SafetyLevel.normal


def escalation_block() -> list[str]:
    """The deterministic escalation block as rendered Markdown lines.

    Always returned verbatim from the constants — never assembled from model
    output. Used by the pipeline when the level is ``urgent``: a heading, a
    blank line, then one bullet per fixed guidance line.
    """
    lines = [f"## {ESCALATION_TITLE}", ""]
    lines.extend(f"- {line}" for line in ESCALATION_LINES)
    return lines


__all__ = [
    "DISCLAIMER",
    "ESCALATION_LINES",
    "ESCALATION_TITLE",
    "LEVEL_LABEL",
    "SUPPORT_NOTE",
    "SafetyLevel",
    "classify",
    "escalation_block",
]
