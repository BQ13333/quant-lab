from quant_lab.notes.obsidian import ObsidianNote, write_note


def test_write_obsidian_note(tmp_path):
    path = write_note(
        tmp_path,
        "30_experiments/test/index.md",
        ObsidianNote(
            title="Test Note",
            note_type="experiment",
            tags=["quant-lab"],
            metadata={"experiment_id": "EXP-1"},
            body="Body text",
        ),
    )

    text = path.read_text()
    assert "type: \"experiment\"" in text
    assert "experiment_id: \"EXP-1\"" in text
    assert "# Test Note" in text
