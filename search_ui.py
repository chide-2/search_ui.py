"""
search_ui.py — OWNED BY: Person 3 (Search Experience)
========================================================

The full search flow end-to-end: the "did you mean" dropdown, running
a search, calling the FDA client (Person 2), calling Gemini to simplify each label
section (Person 4's AITranslator), and saving the result to history
(Person 8's SearchHistory / build_entry).
"""

from __future__ import annotations

import time
from typing import Callable, cast

from ai_translator import AITranslator
from fda_client import FDAClient, get_cached_suggestions, render_recall_banner
from history_stores import SearchHistory, build_entry
from models import (
    AITranslationError,
    FDANetworkError,
    InvalidMedicationNameError,
    Medication,
    MedicationNotFoundError,
)
from st_compat import (
    _StreamlitErrorAPI,
    _StreamlitSpinnerFactoryAPI,
    _StreamlitWarningAPI,
    _StreamlitWriteAPI,
    st,
)


# ---------------------------------------------------------------------------
# "Did you mean" dropdown
# ---------------------------------------------------------------------------

_SUGGESTION_PLACEHOLDER: str = "Did you mean…"


def render_search_suggestions(session_state: dict[str, object], drug_name_input: str) -> None:
    """Render a "did you mean" dropdown under the search box, driven by
    FDAClient.suggest_names(). Picking an option fills the search box
    with the exact FDA-recognized name and immediately triggers a
    search, so the user doesn't have to click Search a second time.

    The dropdown's widget key is derived from the current prefix
    (`suggestion_select::<prefix>`) rather than a single fixed key.
    Streamlit requires a selectbox's stored value to be one of its
    current `options`, and the suggestion list changes on every
    keystroke — a fixed key would raise once a previously-selected
    option disappeared from a new options list. Keying per prefix
    sidesteps that: each prefix gets its own fresh widget state, so
    there's never a stale selection sitting outside the current options.
    """
    cleaned = drug_name_input.strip()
    if len(cleaned) < 2:
        return

    suggestions = get_cached_suggestions(cleaned)
    # Don't show a suggestion that's just an exact echo of what's
    # already typed — that's not a useful "did you mean".
    suggestions = tuple(s for s in suggestions if s.lower() != cleaned.lower())
    if not suggestions:
        return

    dropdown_key = f"suggestion_select::{cleaned.lower()}"
    options = [_SUGGESTION_PLACEHOLDER, *suggestions]

    def _apply_selected_suggestion() -> None:
        selected = cast(str, session_state.get(dropdown_key, _SUGGESTION_PLACEHOLDER))
        if selected and selected != _SUGGESTION_PLACEHOLDER:
            # Setting these here is what lets the text_input (bound to
            # the same session_state key) pick up the chosen name on
            # the rerun Streamlit already triggers after on_change, and
            # auto-search it without a second click.
            session_state["drug_name_input"] = selected
            session_state["trigger_search"] = True

    cast(Callable[..., object], getattr(cast(object, st), "selectbox"))(
        "Did you mean:",
        options,
        key=dropdown_key,
        on_change=_apply_selected_suggestion,
    )


# ---------------------------------------------------------------------------
# Full search tab
# ---------------------------------------------------------------------------

def render_search_tab(history: SearchHistory, session_state: dict[str, object], api_key: str) -> None:
    """Render the drug-lookup search box, results, recall banner, and
    AI-simplified label sections."""
    if "drug_name_input" not in session_state:
        session_state["drug_name_input"] = ""
    if "trigger_search" not in session_state:
        session_state["trigger_search"] = False

    drug_name_input = cast(
        Callable[..., str], getattr(cast(object, st), "text_input")
    )(
        "Enter a medication name",
        placeholder="e.g. ibuprofen",
        key="drug_name_input",
    )

    # Live "did you mean" suggestions as the user types, powered by
    # FDAClient.suggest_names(). Runs on every rerun (i.e. every
    # keystroke that Streamlit picks up), but is cheap thanks to
    # get_cached_suggestions() and openFDA's lightweight count-query mode.
    render_search_suggestions(session_state, drug_name_input)

    search_clicked = cast(
        Callable[..., bool], getattr(cast(object, st), "button")
    )("Search", type="primary")

    if not (search_clicked or session_state.get("trigger_search")):
        return

    # A suggestion click already consumed itself by setting this flag —
    # reset it so a plain rerun later (e.g. clearing history) doesn't
    # re-trigger a search on its own.
    session_state["trigger_search"] = False

    # 1. Validate input
    try:
        drug_name = Medication.validate_name(drug_name_input)
    except InvalidMedicationNameError as exc:
        _ = cast(_StreamlitErrorAPI, cast(object, st)).error(str(exc))
        return

    fda_client = FDAClient()
    medication: Medication | None = None
    recalls: list[dict[str, str]] = []

    # 2. Fetch label info
    with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner(
        f"Looking up '{drug_name}'..."
    ):
        try:
            medication = fda_client.fetch_label(drug_name)
        except MedicationNotFoundError as exc:
            _ = cast(_StreamlitWarningAPI, cast(object, st)).warning(str(exc))
            return
        except FDANetworkError as exc:
            _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                f"Network problem while fetching drug info: {exc}"
            )
            return

        missing = medication.has_missing_fields()
        if missing:
            cast(Callable[[str], object], getattr(cast(object, st), "info"))(
                "Note: the FDA label was missing data for: " + ", ".join(missing)
            )

        # 3. Fetch recalls (non-fatal if it fails)
        try:
            recalls = fda_client.fetch_recalls(medication.generic_name or drug_name)
        except FDANetworkError as exc:
            _ = cast(_StreamlitWarningAPI, cast(object, st)).warning(
                f"Could not check recalls right now: {exc}"
            )
            recalls = []

    cast(Callable[[str], object], getattr(cast(object, st), "subheader"))(
        f"{medication.generic_name.title() or drug_name.title()}"
    )
    if medication.brand_names:
        cast(Callable[[str], object], getattr(cast(object, st), "caption"))(
            "Brand names: " + ", ".join(medication.brand_names)
        )

    render_recall_banner(recalls)

    keywords = medication.extract_warning_keywords()
    if keywords:
        cast(Callable[[str], object], getattr(cast(object, st), "markdown"))(
            "**Key warning phrases (raw extract):**"
        )
        _ = cast(_StreamlitWriteAPI, cast(object, st)).write(
            " • " + "\n • ".join(keywords)
        )

    # 4. Simplify with Gemini
    simplified: dict[str, str] = {}
    sections = {
        "Usage": medication.usage,
        "Dosage & Administration": medication.dosage,
        "Warnings": medication.warnings,
        "Side Effects": medication.side_effects,
    }

    markdown_fn = cast(Callable[[str], object], getattr(cast(object, st), "markdown"))
    write_fn = cast(_StreamlitWriteAPI, cast(object, st))

    if not api_key:
        _ = cast(_StreamlitWarningAPI, cast(object, st)).warning(
            "No Gemini API key provided — showing raw FDA text instead of "
            + "the simplified version."
        )
        for label, text in sections.items():
            markdown_fn(f"### {label}")
            write_fn.write(text or "_No data available._")
    else:
        translator = AITranslator(api_key=api_key)
        for label, text in sections.items():
            markdown_fn(f"### {label}")
            try:
                with cast(_StreamlitSpinnerFactoryAPI, cast(object, st)).spinner(
                    f"Simplifying '{label}'..."
                ):
                    simple_text = translator.simplify(text, label)
                write_fn.write(simple_text)
                simplified[label] = simple_text
            except AITranslationError as exc:
                _ = cast(_StreamlitErrorAPI, cast(object, st)).error(
                    f"Could not simplify this section: {exc}"
                )
                write_fn.write(text or "_No data available._")
            time.sleep(0.2)  # gentle pacing between API calls

    # 5. Save to history
    entry = build_entry(drug_name, medication, recalls, simplified)
    history.add(entry)
    cast(Callable[[str], object], getattr(cast(object, st), "toast"))(
        "Search saved to history."
    )
