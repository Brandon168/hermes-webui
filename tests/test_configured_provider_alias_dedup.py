"""Regression coverage for provider-qualified alias duplication in the picker.

The server's ``configured_model_badges`` map can carry more than one key for a
single configured entry: the @-bound form (``@commandcode:deepseek/model``) and
a provider-qualified alias (``commandcode/deepseek/model``). The CONFIGURED
section's semantic-key dedupe must collapse both into one row.

Failure mode (this test fails before the fix): ``_normalizeConfiguredModelKey``
peels one slash segment for the alias but two layers for the @-bound form, so
the two keys differ and the picker renders the same model twice — displaying the
decorated alias alongside the safe @-bound form.

Tests run the live JS functions via Node so drift is caught immediately.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.resolve()
UI_JS_PATH = REPO_ROOT / "static" / "ui.js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not on PATH")


_DRIVER = r"""
const fs = require('fs');
const ui = fs.readFileSync(process.argv[2], 'utf8');
function extractFunc(name) {
  const re = new RegExp('function\\s+' + name + '\\s*\\(');
  const start = ui.search(re);
  if (start < 0) throw new Error(name + ' not found');
  let i = ui.indexOf('{', start); let depth = 1; i++;
  while (depth > 0 && i < ui.length) {
    if (ui[i] === '{') depth++;
    else if (ui[i] === '}') depth--;
    i++;
  }
  return ui.slice(start, i);
}
eval(extractFunc('_normalizeConfiguredModelKey'));
let _hasPriority = false;
try { eval(extractFunc('_configuredDisplayPriority')); _hasPriority = typeof _configuredDisplayPriority === 'function'; } catch (e) { _hasPriority = false; }
const cases = JSON.parse(process.argv[3]);
const result = cases.map(c => ({
  key: _normalizeConfiguredModelKey(c.value, c.provider || ''),
  priority: _hasPriority ? _configuredDisplayPriority({value: c.value, badge: {provider: c.provider || ''}}) : -1,
}));
process.stdout.write(JSON.stringify(result));
"""


def _run(tmp_path, cases):
    driver = tmp_path / "driver.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    assert NODE is not None
    result = subprocess.run(
        [NODE, str(driver), str(UI_JS_PATH), json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


BOUND = "@commandcode:deepseek/deepseek-v4.1-flash"
ALIAS = "commandcode/deepseek/deepseek-v4.1-flash"


def test_provider_alias_and_bound_form_share_one_semantic_key(tmp_path):
    """The dedupe map must see one key for both spellings of the entry."""
    out = _run(
        tmp_path,
        [
            {"value": BOUND, "provider": "commandcode"},
            {"value": ALIAS, "provider": "commandcode"},
        ],
    )
    assert out[0]["key"] == out[1]["key"], (
        f"alias {ALIAS!r} must collapse with bound form {BOUND!r}, "
        f"got keys {out[0]['key']!r} vs {out[1]['key']!r}"
    )


def test_provider_peel_only_strips_matching_provider(tmp_path):
    """A leading segment equal to another provider must not be peeled."""
    out = _run(
        tmp_path,
        [
            {"value": "vendor_b/deepseek/deepseek-v4-pro", "provider": "commandcode"},
            {"value": "commandcode/deepseek/deepseek-v4-pro", "provider": "commandcode"},
        ],
    )
    assert out[0]["key"] != out[1]["key"], (
        "provider peel must not strip segments that do not match the entry provider"
    )


def test_no_provider_context_keeps_legacy_normalization(tmp_path):
    """One-argument calls (tests, badge lookups) keep pre-existing behavior."""
    out = _run(tmp_path, [{"value": "vendor_a/deepseek/deepseek-v4-pro", "provider": ""}])
    assert out[0]["key"] == "deepseek/deepseek.v4.pro"


def test_bound_form_wins_display_over_provider_alias(tmp_path):
    """When spellings collide, the safe @-bound form must be displayed."""
    out = _run(
        tmp_path,
        [
            {"value": BOUND, "provider": "commandcode"},
            {"value": ALIAS, "provider": "commandcode"},
        ],
    )
    assert out[0]["priority"] > out[1]["priority"], (
        "@-bound form must outrank the provider-qualified alias for display"
    )
