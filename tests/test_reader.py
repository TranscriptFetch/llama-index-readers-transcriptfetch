from __future__ import annotations

import json

import httpx
import pytest
import respx
from llama_index.core.readers.base import BaseReader
from llama_index.core.schema import MetadataMode

from llama_index.readers.transcriptfetch import TranscriptFetchReader

BASE = "https://api.test"
CHUNKS = [
    {"index": 0, "text": "first", "start": 0.0, "end": 30.5, "tokens": 400,
     "url": "https://www.youtube.com/watch?v=a&t=0s"},
    {"index": 1, "text": "second", "start": 28.0, "end": 61.0, "tokens": 380,
     "url": "https://www.youtube.com/watch?v=a&t=28s"},
]


def _batch(results: list) -> dict:
    return {"ok": True, "data": {"kind": "transcript_batch", "results": results}}


def _item(vid: str, **extra) -> dict:
    return {"video_id": vid, "outcome": "ok", "url": f"https://www.youtube.com/watch?v={vid}",
            "title": f"Video {vid}", "channel": "Example", "source": "captions",
            "text": f"text of {vid}", **extra}


def test_is_a_reader() -> None:
    assert BaseReader.__name__ in [c.__name__ for c in TranscriptFetchReader.__mro__]


@respx.mock
def test_load_per_video() -> None:
    respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([_item("a"), _item("b")]))
    )
    docs = TranscriptFetchReader(api_key="k", base_url=BASE).load_data(["a", "b"])
    assert [d.text for d in docs] == ["text of a", "text of b"]
    assert docs[0].id_ == "a"
    assert docs[0].metadata["platform"] == "youtube"


@respx.mock
def test_chunks_keep_links_out_of_embeddings() -> None:
    route = respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([_item("a", chunks=CHUNKS)]))
    )
    reader = TranscriptFetchReader(api_key="k", base_url=BASE, chunks=True, chunk_tokens=300)
    docs = reader.load_data("a")
    assert [d.text for d in docs] == ["first", "second"]
    d = docs[1]
    assert d.metadata["source"] == "https://www.youtube.com/watch?v=a&t=28s"
    assert d.id_ == "a:1"
    embedded = d.get_content(metadata_mode=MetadataMode.EMBED)
    assert "Video a" in embedded and "t=28s" not in embedded
    assert json.loads(route.calls.last.request.content)["chunks"] == {
        "max_tokens": 300, "overlap_tokens": 50,
    }


@respx.mock
def test_search_platform_is_sent() -> None:
    search = respx.post(f"{BASE}/api/v2/transcripts/search").mock(
        return_value=httpx.Response(200, json={"ok": True, "data": {"kind": "video_list", "videos": [
            {"videoId": "1", "url": "https://www.tiktok.com/@x/video/1"},
        ]}})
    )
    respx.post(f"{BASE}/api/v2/transcripts/batch").mock(
        return_value=httpx.Response(200, json=_batch([
            {**_item("1"), "url": "https://www.tiktok.com/@x/video/1"},
        ]))
    )
    docs = TranscriptFetchReader(api_key="k", base_url=BASE).load_data(
        search="sourdough", search_platform="tiktok", max_videos=1
    )
    assert json.loads(search.calls.last.request.content)["platform"] == "tiktok"
    assert docs[0].metadata["platform"] == "tiktok"


def test_one_source_only() -> None:
    with pytest.raises(ValueError):
        TranscriptFetchReader(api_key="k").load_data("a", channel="@b")
