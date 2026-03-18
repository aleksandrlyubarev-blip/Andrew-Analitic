from core.romeo_phd import SYSTEM_PROMPT


def test_prompt_preserves_pedagogical_identity() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "expert educator" in prompt
    assert "concrete examples" in prompt
    assert "analogies" in prompt
    assert "learning paths" in prompt
    assert "pedagogically sound" in prompt


def test_prompt_keeps_markdown_contract() -> None:
    assert "Format every response in Markdown:" in SYSTEM_PROMPT
    assert "**In Practice**" in SYSTEM_PROMPT
    assert "inline code" in SYSTEM_PROMPT
    assert "headers (## / ###)" in SYSTEM_PROMPT


def test_prompt_preserves_andrew_handoff() -> None:
    assert 'For the computation, try asking Andrew:' in SYSTEM_PROMPT
    assert "Andrew Swarm handles computational analysis" in SYSTEM_PROMPT
