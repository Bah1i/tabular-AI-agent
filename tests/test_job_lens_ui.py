from pathlib import Path


def test_job_lens_ui_hides_empty_blocks_and_shows_lens_cycle():
    template = Path("app/templates/job.html").read_text(encoding="utf-8")

    assert "function isBlankLensValue" in template
    assert "text === '-'" in template
    assert "text === '—'" in template
    assert "renderedBlocks.length" in template
    assert "0. Lens cycle" in template
    assert "Forward transform" in template
    assert "Backward propagation" in template
    assert "Consistency restoration" in template
    assert "0. Summary signals" in template
    assert "function renderHybridLensText" in template
    assert "function renderCalibratedConfidenceText" in template
    assert "Hybrid lens-backed status" in template
    assert "Calibrated confidence" in template
    assert "Delta discovery" in template
    assert "Putback target" in template
    assert "Putback mode" in template
    assert "Amendment policy" in template
    assert "Runtime law checks" in template
    assert "Restoration level" in template
    assert "function renderTypedDeltaText" in template
    assert "function renderPutbackPolicyText" in template
    assert "function renderPutbackModeText" in template
    assert "function renderLensLawChecksText" in template
    assert "function renderRestorationStateText" in template
    assert "function isDuplicateLensSummarySection" in template
    assert "title.startsWith('0.')" in template
    assert "title.startsWith('1.')" in template
    assert "node.remove()" in template
    assert "function buildCompactLensSummary" in template
    assert "lens-compact-section" in template
    assert "lens-compact-field" in template
    assert "Technical details" in template
    assert "lens-overview" in template
