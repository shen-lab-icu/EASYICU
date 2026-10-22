"""Every cohort step of an extraction has a Chinese progress line.

The hospital walk (10-22) showed 「Cohort resolved」 on the Chinese page:
the job emitted only an English ``message``. The page shows ``message_zh``
when a progress event carries one.
"""

from __future__ import annotations

from easyicu.webserver import dataio


def test_each_cohort_stage_has_a_chinese_line():
    english = dataio._COHORT_PROGRESS_MESSAGES
    chinese = dataio._COHORT_PROGRESS_MESSAGES_ZH
    assert set(chinese) == set(english)
    for stage, text in chinese.items():
        assert text.strip(), stage
        assert not any("a" <= ch.lower() <= "z" for ch in text.replace("ICU", "").replace("ICD", "")), text
