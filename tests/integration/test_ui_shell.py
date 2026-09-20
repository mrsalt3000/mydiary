"""Integration tests for the UI shell and the theme (task T16a).

Covers the first stage of the mockup redesign: the copied reference mockup,
the single token system, the manual theme switch (system setting by default,
applied before the first paint), the three-column shell, the sidebar (real
links, real counters, no dead entries) and the fact that every existing page
is rendered inside that shell.

Most assertions are *static contracts* over the templates, the stylesheet and
the client script — honest about what they prove: that the markup, the tokens
and the wiring are what the design requires. Whether the toggle really
switches without a flash, and how the layout collapses at narrow widths, are
runtime properties verified in a browser, not here.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from diary.api.routes.messages import get_session
from diary.api.routes.pages import pages_router
from diary.core.settings import reload_settings
from diary.db.models import JobStatus, JobType, MessageKind
from diary.db.repositories import JobRepository, MessageRepository
from diary.domain.exports.service import export_week
from diary.main import app

#: Repository root: tests/integration/test_ui_shell.py -> parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = REPO_ROOT / "src/diary/templates"
STATIC_DIR = REPO_ROOT / "src/diary/static"
MOCKUP = REPO_ROOT / "docs/mockups/mydiary-ui-mockup.html"

SUNDAY = date(2026, 9, 13)
SUNDAY2 = date(2026, 9, 6)

#: Pages that must render inside the shell (all existing UI routes).
PAGE_PATHS = ("/", "/exports", "/processing", "/privacy")

#: Tokens the mockup defines and the stylesheet must therefore carry; every
#: one of them is used by the shell, so none is decorative.
MOCKUP_TOKENS = (
    "--bg",
    "--surface",
    "--surface-muted",
    "--surface-hover",
    "--text",
    "--text-muted",
    "--border",
    "--border-strong",
    "--accent",
    "--accent-hover",
    "--accent-soft",
    "--accent-contrast",
    "--success",
    "--success-soft",
    "--warning",
    "--warning-soft",
    "--danger",
    "--danger-soft",
    "--shadow",
    "--radius-sm",
    "--radius-md",
    "--radius-lg",
    "--sidebar-w",
    "--context-w",
)


@pytest.fixture(autouse=True)
def _isolate_app_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the app data root at a per-test directory and isolate the env."""
    monkeypatch.delenv("APP_DATA_ROOT", raising=False)
    monkeypatch.setenv("APP_DATA_ROOT", str(tmp_path / "diary-data"))
    reload_settings()


@contextmanager
def _client_bound_to(session_factory) -> Iterator[TestClient]:
    """A TestClient whose ``get_session`` dependency serves ``session_factory``."""

    def _override_session() -> Iterator[Session]:
        with session_factory() as current_session:
            yield current_session

    app.dependency_overrides[get_session] = _override_session
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.fixture
def api_client(session_factory) -> Iterator[TestClient]:
    with _client_bound_to(session_factory) as client:
        yield client


def _create_export(session: Session, sunday: date) -> object:
    """One real weekly export (row + file) with a synthetic message."""
    monday = sunday - timedelta(days=6)
    MessageRepository(session).create(
        kind=MessageKind.text,
        body_text="Синтетическая запись для теста оболочки.",
        created_at=datetime(monday.year, monday.month, monday.day, 9, 0, tzinfo=timezone.utc),
    )
    session.commit()
    return export_week(session, sunday_date=sunday)


def _sidebar(html: str) -> str:
    match = re.search(r'<aside class="sidebar".*?</aside>', html, re.DOTALL)
    assert match is not None, "the shell must render the sidebar"
    return match.group(0)


def _nav_item(sidebar: str, href: str) -> str:
    match = re.search(
        r'<a class="nav-item[^"]*"\s*\n?\s*href="' + re.escape(href) + r'"[^>]*>',
        sidebar,
    )
    assert match is not None, f"no navigation item for {href}"
    return match.group(0)


# ---------------------------------------------------------------------------
# the reference mockup
# ---------------------------------------------------------------------------


def test_reference_mockup_is_kept_in_the_repository() -> None:
    """T16a copies the mockup into the repo so the acceptance compares against
    the artefact, not against memory."""
    text = MOCKUP.read_text(encoding="utf-8")
    for marker in (
        "--sidebar-w: 244px",
        "--context-w: 332px",
        'data-theme="dark"',
        'class="app-shell"',
        "@media (max-width: 1080px)",
        "@media (max-width: 700px)",
    ):
        assert marker in text, marker
    # The mockup itself pulls in no external resource (privacy rule, AGENTS §7).
    assert re.search(r"https?://", text) is None


# ---------------------------------------------------------------------------
# one token system
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    return re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)


def _declarations(block: str) -> dict[str, str]:
    """``name: value`` pairs of a CSS block, comments removed."""
    return {
        name.strip(): value.strip()
        for name, value in re.findall(r"([-\w]+)\s*:\s*([^;]+);", _strip_comments(block))
    }


def _css() -> str:
    return (STATIC_DIR / "css/app.css").read_text(encoding="utf-8")


def test_stylesheet_carries_the_mockup_token_system() -> None:
    css = _css()
    for token in MOCKUP_TOKENS:
        assert f"{token}:" in css, token
    assert "--sidebar-w: 244px" in css
    assert "--context-w: 332px" in css


def test_the_old_palette_names_are_migrated_not_aliased() -> None:
    """One system only: the previous ``--color-*`` names are gone entirely, so
    no two competing token sets can exist side by side."""
    css = _strip_comments(_css())
    assert "--color-" not in css
    assert "shadow-card" not in css


def test_light_and_dark_declare_the_same_token_set() -> None:
    css = _css()
    light = re.search(r"\n:root \{(.*?)\n\}", css, re.DOTALL)
    media = re.search(
        r"@media \(prefers-color-scheme: dark\)\s*\{\s*"
        r':root:not\(\[data-theme="light"\]\)\s*\{(.*?)\n\s*\}',
        css,
        re.DOTALL,
    )
    explicit = re.search(r':root\[data-theme="dark"\]\s*\{(.*?)\n\}', css, re.DOTALL)
    assert light is not None and media is not None and explicit is not None

    light_declarations = _declarations(light.group(1))
    dark_declarations = _declarations(media.group(1))
    # The dark palettes are duplicated on purpose (CSS cannot share one block
    # between a media query and an attribute selector) — this is the guard
    # that stops them from drifting apart.
    assert dark_declarations == _declarations(explicit.group(1))
    # Layout and radii do not change with the theme; every colour does.
    theme_independent = {"--sidebar-w", "--context-w", "--radius-sm", "--radius-md", "--radius-lg"}
    assert set(dark_declarations) == set(light_declarations) - theme_independent
    assert light_declarations["color-scheme"] == "light"
    assert dark_declarations["color-scheme"] == "dark"


def test_shell_breakpoints_follow_the_mockup() -> None:
    css = _css()
    narrow = re.search(r"@media \(max-width: 1080px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    phone = re.search(r"@media \(max-width: 700px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert narrow is not None and phone is not None
    narrow_block = _strip_comments(narrow.group(1))
    phone_block = _strip_comments(phone.group(1))

    # <=1080px: two columns, the context panel disappears.
    assert "grid-template-columns: var(--sidebar-w) minmax(0, 1fr)" in narrow_block
    assert re.search(r"\.context-panel\s*\{[^}]*display: none", narrow_block)
    # <=700px: the sidebar disappears and the column collapses to zero width.
    assert "--sidebar-w: 0px" in phone_block
    assert re.search(r"\.sidebar\s*\{[^}]*display: none", phone_block)

    mockup = MOCKUP.read_text(encoding="utf-8")
    for width in ("1080px", "700px"):
        assert f"@media (max-width: {width})" in mockup


def test_three_column_shell_uses_the_mockup_width_tokens() -> None:
    css = _strip_comments(_css())
    assert re.search(
        r"\.app-shell\s*\{[^}]*grid-template-columns:\s*var\(--sidebar-w\) minmax\(0, 1fr\);",
        css,
    )
    assert re.search(
        r"\.app-shell--with-context\s*\{[^}]*grid-template-columns:\s*"
        r"var\(--sidebar-w\) minmax\(0, 1fr\) var\(--context-w\);",
        css,
    )


# ---------------------------------------------------------------------------
# theme: applied before the first paint, switchable by hand
# ---------------------------------------------------------------------------


def test_theme_script_is_inline_in_head_before_the_stylesheet(api_client: TestClient) -> None:
    """A stored choice must be applied before the first paint, otherwise a dark
    page would flash light."""
    html = api_client.get("/").text
    head = html[html.index("<head>") : html.index("</head>")]
    script_start = head.index("mydiary-theme")
    assert head.index("<script>") < script_start
    assert "document.documentElement.setAttribute" in head
    assert script_start < head.index('href="/static/css/app.css"')
    # No ``defer``/``src`` on that script: it must run synchronously.
    inline = head[head.index("<script>") : head.index("</script>")]
    assert "src=" not in inline and "defer" not in inline


def test_no_stored_choice_leaves_the_system_setting_in_charge() -> None:
    base = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
    # Nothing is written to <html> unless localStorage holds a valid choice:
    # the CSS prefers-color-scheme block then decides (also without JS).
    script = base[base.index("<script>") : base.index("</script>")]
    assert 'getItem("mydiary-theme")' in script
    assert 'stored === "dark" || stored === "light"' in script
    css = _css()
    assert "@media (prefers-color-scheme: dark)" in css


def test_theme_toggle_is_reachable_and_names_its_state(api_client: TestClient) -> None:
    html = api_client.get("/").text
    toggles = re.findall(r'<button type="button" class="theme-toggle"[^>]*>', html)
    # One in the sidebar, one in the topbar: the mockup hides the whole sidebar
    # at <=700px, which would otherwise make the switch unreachable there.
    assert len(toggles) == 2
    assert html.count('data-action="toggle-theme"') == 2
    # The state is announced, not only painted.
    assert html.count('data-role="theme-name"') == 2
    assert html.count('aria-live="polite"') == 2
    assert ">Светлая<" in html
    # Native buttons: keyboard operable and focusable without extra work.
    assert 'class="theme-toggle" data-action="toggle-theme"' in html
    script = (STATIC_DIR / "js/app.js").read_text(encoding="utf-8")
    assert "setItem(STORAGE_KEY, next)" in script
    assert 'setAttribute("data-theme", theme)' in script
    assert "prefers-color-scheme: dark" in script


# ---------------------------------------------------------------------------
# sidebar: real links, real counters, nothing dead
# ---------------------------------------------------------------------------


def test_sidebar_links_point_at_routes_that_exist(api_client: TestClient) -> None:
    """Every navigation entry is a real page — and every one of them answers."""
    sidebar = _sidebar(api_client.get("/").text)
    hrefs = re.findall(r'href="([^"]+)"', sidebar)
    assert hrefs == ["/", "/exports", "/processing", "/privacy"]
    route_paths = {getattr(route, "path", None) for route in pages_router.routes}
    for href in hrefs:
        assert href in route_paths, href
        assert api_client.get(href).status_code == 200, href


def test_sidebar_omits_mockup_items_that_have_no_page(api_client: TestClient) -> None:
    """No dead entries: mockup navigation for which no route exists is not
    rendered, and the search icon stays out until there is a search endpoint."""
    html = api_client.get("/").text
    for missing in ("Хронология", "Настройки"):
        assert missing not in html, missing
    assert "Поиск" not in html
    # The topbar carries no action of its own in T16a (no export button: the
    # export form lives on /exports).
    assert "Экспорт недели" not in html


def test_sidebar_counters_come_from_the_database(api_client: TestClient, session: Session) -> None:
    """Two exports and two in-flight jobs are counted as such; the third job is
    terminal and is not counted. No number is invented."""
    _create_export(session, SUNDAY)
    _create_export(session, SUNDAY2)
    repository = JobRepository(session)
    repository.create(job_type=JobType.transcribe_audio, status=JobStatus.pending)
    repository.create(job_type=JobType.normalize, status=JobStatus.running)
    repository.create(job_type=JobType.summarize, status=JobStatus.succeeded)
    session.commit()

    html = api_client.get("/").text
    assert '<span class="visually-hidden">всего: </span>2</span>' in html
    assert '<span class="visually-hidden">задач в работе: </span>2</span>' in html


def test_sidebar_counters_are_zero_on_an_empty_database(api_client: TestClient) -> None:
    html = api_client.get("/").text
    assert '<span class="visually-hidden">всего: </span>0</span>' in html
    assert '<span class="visually-hidden">задач в работе: </span>0</span>' in html


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_active_nav_item_matches_the_page(api_client: TestClient, path: str) -> None:
    sidebar = _sidebar(api_client.get(path).text)
    active = [item for item in re.findall(r"<a class=\"nav-item[^\"]*\"[^>]*>", sidebar)]
    marked = [item for item in active if "nav-item--active" in item]
    assert len(marked) == 1, sidebar
    assert 'aria-current="page"' in marked[0]
    assert f'href="{path}"' in marked[0]


# ---------------------------------------------------------------------------
# every existing page renders inside the shell
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", PAGE_PATHS)
def test_pages_render_inside_the_shell(api_client: TestClient, path: str) -> None:
    response = api_client.get(path)
    assert response.status_code == 200
    html = response.text
    # The class may carry the "with-context" modifier (the day page does, T16c).
    assert re.search(r'class="app-shell[\s"]', html)
    assert '<aside class="sidebar"' in html
    assert 'class="topbar"' in html
    assert "<h1>" in html
    # The legacy in-content navigation is gone: the sidebar is the navigation.
    assert 'class="page-nav"' not in html


def test_export_pages_render_inside_the_shell(api_client: TestClient, session: Session) -> None:
    record = _create_export(session, SUNDAY)
    export_id = str(record.id)  # type: ignore[attr-defined]
    for path in (
        f"/exports/{export_id}",
        f"/exports/{export_id}/normalize",
        f"/exports/{export_id}/summary",
        f"/exports/{export_id}/wellbeing",
    ):
        response = api_client.get(path)
        assert response.status_code == 200, path
        html = response.text
        assert 'class="app-shell"' in html, path
        assert '<aside class="sidebar"' in html, path
        assert 'class="page-nav"' not in html, path
        # The export pages belong to the "exports" section of the sidebar.
        assert "nav-item--active" in _sidebar(html), path


def test_privacy_page_renders_without_the_database_and_without_counters() -> None:
    """The privacy page stays database-free (Task 18): it highlights its own
    item but renders no counters, because it never opens the database.
    """
    with TestClient(app) as client:
        response = client.get("/privacy")
    assert response.status_code == 200
    sidebar = _sidebar(response.text)
    assert 'class="count"' not in sidebar
    assert "nav-item--active" in sidebar
    assert 'aria-current="page"' in sidebar


def test_narrow_screen_navigation_keeps_every_section_reachable(
    api_client: TestClient,
) -> None:
    """T16b finding from the T16a acceptance: at <=700px the sidebar is hidden,
    which left the narrow layout with no way to change the page.

    The compact menu is a native ``<details>`` disclosure in the topbar: it
    needs no script, is operable from the keyboard (Enter/Space on the
    ``<summary>``, Tab through the links) and is hidden *above* the
    breakpoint, so it never duplicates the sidebar for wide layouts. It carries
    the same links as the sidebar — one included list, so the two copies
    cannot drift apart.
    """
    html = api_client.get("/").text
    menu = re.search(r'<details class="compact-nav">.*?</details>', html, re.DOTALL)
    assert menu is not None, "the shell must carry the compact navigation menu"
    markup = menu.group(0)
    # Native disclosure: no script, keyboard operable, named for assistive tech.
    assert "<summary" in markup and "Разделы" in markup
    assert 'class="compact-nav__menu"' in markup
    assert 'aria-label="Разделы (компактное меню)"' in markup

    hrefs = re.findall(r'href="([^"]+)"', markup)
    assert hrefs == ["/", "/exports", "/processing", "/privacy"]
    route_paths = {getattr(route, "path", None) for route in pages_router.routes}
    for href in hrefs:
        assert href in route_paths, href
        assert api_client.get(href).status_code == 200, href

    # Only one item is highlighted, exactly as in the sidebar.
    assert markup.count("nav-item--active") == 1
    assert 'aria-current="page"' in markup

    css = _css()
    phone = re.search(r"@media \(max-width: 700px\)\s*\{(.*?)\n\}", css, re.DOTALL)
    assert phone is not None
    phone_block = _strip_comments(phone.group(1))
    assert re.search(r"\.sidebar\s*\{[^}]*display: none", phone_block)
    assert re.search(r"\.compact-nav\s*\{[^}]*display: block", phone_block)
    # Hidden above the breakpoint: the sidebar is the navigation there.
    base_rule = re.search(r"\n\.compact-nav\s*\{(.*?)\n\}", css, re.DOTALL)
    assert base_rule is not None
    assert "display: none" in base_rule.group(1)

    # No script is involved in the disclosure at all.
    script = (STATIC_DIR / "js" / "app.js").read_text(encoding="utf-8")
    assert "compact-nav" not in script


def test_exactly_the_day_page_fills_the_context_panel() -> None:
    """The shell's third column is opt-in: a page must set ``context_panel``
    *and* fill the ``context`` block, so a page without context does not
    reserve an empty 332px column.

    Since T16c exactly one page does — the day page, whose processing panel is
    that task (its content is covered by ``test_context_panel.py``). The other
    pages must stay two-column.
    """
    base = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
    assert "{% if context_panel %}" in base
    assert 'class="context-panel"' in base

    chat = (TEMPLATES_DIR / "chat.html").read_text(encoding="utf-8")
    assert "{% block context %}" in chat
    assert "_context_panel.html" in chat

    for name in (
        "exports.html",
        "processing.html",
        "privacy.html",
        "export_detail.html",
        "export_compare.html",
        "export_summary.html",
        "export_wellbeing.html",
    ):
        source = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        assert "context_panel" not in source, name
        assert "{% block context %}" not in source, name


def test_shell_pages_set_their_own_title_and_subtitle() -> None:
    """Each page names itself in the topbar; the subtitle is optional."""
    for name in (
        "chat.html",
        "exports.html",
        "processing.html",
        "privacy.html",
        "export_detail.html",
        "export_compare.html",
        "export_summary.html",
        "export_wellbeing.html",
    ):
        source = (TEMPLATES_DIR / name).read_text(encoding="utf-8")
        assert "{% block page_title %}" in source, name
        assert '{% extends "base.html" %}' in source, name
