from document_qa.document_qa_engine import TextMerger


def test_merge_passages_small_chunk():
    merger = TextMerger()

    passages = [
        {"text": "The quick brown fox jumps over the tree", "coordinates": "1"},
        {"text": "and went straight into the mouth of a bear.", "coordinates": "2"},
        {"text": "The color of the colors is a color with colors", "coordinates": "3"},
        {"text": "the main colors are not the colorw we show", "coordinates": "4"},
    ]
    new_passages = merger.merge_passages(passages, chunk_size=10, tolerance=0)

    assert len(new_passages) == 4
    assert new_passages[0]["coordinates"] == "1"
    assert new_passages[0]["text"] == "The quick brown fox jumps over the tree"

    assert new_passages[1]["coordinates"] == "2"
    assert new_passages[1]["text"] == "and went straight into the mouth of a bear."

    assert new_passages[2]["coordinates"] == "3"
    assert new_passages[2]["text"] == "The color of the colors is a color with colors"

    assert new_passages[3]["coordinates"] == "4"
    assert new_passages[3]["text"] == "the main colors are not the colorw we show"


def test_merge_passages_big_chunk():
    merger = TextMerger()

    passages = [
        {"text": "The quick brown fox jumps over the tree", "coordinates": "1"},
        {"text": "and went straight into the mouth of a bear.", "coordinates": "2"},
        {"text": "The color of the colors is a color with colors", "coordinates": "3"},
        {"text": "the main colors are not the colorw we show", "coordinates": "4"},
    ]
    new_passages = merger.merge_passages(passages, chunk_size=20, tolerance=0)

    assert len(new_passages) == 2
    assert new_passages[0]["coordinates"] == "1;2"
    assert new_passages[0]["text"] == "The quick brown fox jumps over the tree and went straight into the mouth of a bear."

    assert new_passages[1]["coordinates"] == "3;4"
    assert (
        new_passages[1]["text"] == "The color of the colors is a color with colors the main colors are not the colorw we show"
    )
