"""The login page tells a clinician on IE, 360 compatibility mode or an old core what to do.

The workspace needs Chrome/Edge 105 or later. The note is visible by default
and hidden only by an @supports rule that such browsers do not understand, so
it needs no script (the gateway CSP allows none).
"""
import re


def test_the_note_shows_unless_the_browser_understands_the_hiding_rule(client):
    page = client.get("/login").text
    style = page[page.index("<style>"):page.index("</style>")]

    note = re.search(r"\.browser-note \{([^}]*)\}", style).group(1)
    assert "display" not in note and "var(" not in note
    assert "@supports selector(:has(a)) { .browser-note { display: none; } }" in style
    assert style.count(".browser-note { display: none; }") == 1


def test_the_note_names_the_browsers_and_the_mode_switch_before_the_form(client):
    page = client.get("/login").text
    note_at = page.index('data-browser-note')
    assert note_at < page.index("<form")
    note = page[note_at:page.index("</div>", note_at)]
    for words in ("Chrome", "Edge", "360", "极速模式"):
        assert words in note
