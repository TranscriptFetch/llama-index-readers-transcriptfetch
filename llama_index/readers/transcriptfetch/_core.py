"""Shared fetching logic for the TranscriptFetch document loaders.

Both the LangChain loader and the LlamaIndex reader vendor this module
unchanged: it turns "these videos / this channel / this playlist / this
search" into a stream of :class:`Record` (one per video, or one per chunk),
and the framework wrapper only maps a record onto its own ``Document`` type.

Transcripts are fetched with the batch endpoint, 50 at a time. Videos without
captions are transcribed from their audio when ``mode`` allows it; the API
answers those with a job, which is polled here until it finishes, so the
caller always gets text or a logged skip, never a half-done item.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Literal

from transcriptfetch import TranscriptFetch
from transcriptfetch.errors import TranscriptFetchError

logger = logging.getLogger(__name__)

Mode = Literal["auto", "captions"]
BATCH_SIZE = 50
PAGE_SIZE = 50


@dataclass
class Record:
    """One unit of text plus the metadata a framework Document carries."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = ""


@dataclass
class FetchOptions:
    videos: Sequence[str] = ()
    channel: str | None = None
    playlist: str | None = None
    search: str | None = None
    search_platform: str | None = None
    max_videos: int = 25
    mode: Mode = "auto"
    chunks: bool = False
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    wait_for_audio: bool = True
    audio_timeout: float = 900.0
    poll_interval: float = 5.0
    continue_on_failure: bool = True


def normalize_videos(videos: str | Iterable[str] | None) -> list[str]:
    if videos is None:
        return []
    if isinstance(videos, str):
        return [videos]
    return [v for v in videos if v]


def validate(opts: FetchOptions) -> None:
    sources = sum(bool(x) for x in (opts.videos, opts.channel, opts.playlist, opts.search))
    if sources != 1:
        raise ValueError("Pass exactly one of: videos, channel, playlist, search.")
    if opts.mode not in ("auto", "captions"):
        raise ValueError('mode must be "auto" or "captions".')
    if opts.max_videos < 1:
        raise ValueError("max_videos must be at least 1.")
    if opts.chunks and not 100 <= opts.chunk_tokens <= 2000:
        raise ValueError("chunk_tokens must be between 100 and 2000.")
    if opts.chunks and not 0 <= opts.chunk_overlap < min(opts.chunk_tokens, 201):
        raise ValueError("chunk_overlap must be 0 to 200 and smaller than chunk_tokens.")


def _targets(client: TranscriptFetch, opts: FetchOptions) -> list[str]:
    """The video URLs (or IDs) to fetch, capped at max_videos for listings."""
    if opts.videos:
        return list(opts.videos)
    tr = client.transcripts
    page = min(PAGE_SIZE, opts.max_videos)
    if opts.channel:
        rows = tr.iter_channel(opts.channel, limit=page)
    elif opts.playlist:
        rows = tr.iter_playlist(opts.playlist, limit=page)
    else:
        rows = tr.iter_search(opts.search or "", platform=opts.search_platform, limit=page)
    return [row.url or row.video_id for row in islice(rows, opts.max_videos)]


def _platform(url: str) -> str | None:
    """Batch results carry no platform field, so read it off the URL."""
    host = url.split("://", 1)[-1].split("/", 1)[0].lower()
    for needle, name in (("youtu", "youtube"), ("tiktok", "tiktok"), ("instagram", "instagram")):
        if needle in host:
            return name
    return None


def _base_metadata(item: Any, requested: str) -> dict[str, Any]:
    url = item.url or requested
    meta = {
        "source": url,
        "video_id": item.video_id or None,
        "platform": getattr(item, "platform", None) or _platform(url),
        "title": item.title,
        "channel": item.channel,
        "duration": item.duration,
        "language": item.language,
        "transcript_source": item.source,
    }
    return {k: v for k, v in meta.items() if v is not None}


def _records(item: Any, requested: str, chunked: bool) -> Iterator[Record]:
    meta = _base_metadata(item, requested)
    vid = meta.get("video_id") or requested
    chunks = getattr(item, "chunks", None) or []
    if chunked and chunks:
        video_url = meta.get("source")
        for c in chunks:
            m = dict(meta)
            m.update({"video_url": video_url, "chunk_index": c.index, "tokens": c.tokens})
            if c.start is not None:
                m["start"] = c.start
            if c.end is not None:
                m["end"] = c.end
            if c.url:
                m["source"] = c.url  # a link to the moment this chunk starts
            yield Record(text=c.text, metadata=m, id=f"{vid}:{c.index}")
        return
    if item.text:
        yield Record(text=item.text, metadata=meta, id=str(vid))


def _fail(opts: FetchOptions, video: str, why: str) -> None:
    if not opts.continue_on_failure:
        raise TranscriptFetchError(f"{video}: {why}")
    logger.warning("TranscriptFetch: skipped %s (%s)", video, why)


def _await_job(client: TranscriptFetch, opts: FetchOptions, video: str, job_id: str) -> Any:
    """Poll an audio-transcription job until it finishes or times out."""
    deadline = time.monotonic() + opts.audio_timeout
    while True:
        t = client.transcripts.job(job_id)
        if t.status != "processing":
            return t
        if time.monotonic() >= deadline:
            _fail(opts, video, f"audio transcription still running after {opts.audio_timeout:.0f}s")
            return None
        time.sleep(opts.poll_interval)


def iter_records(client: TranscriptFetch, opts: FetchOptions) -> Iterator[Record]:
    validate(opts)
    chunk_opts: Any = (
        {"max_tokens": opts.chunk_tokens, "overlap_tokens": opts.chunk_overlap}
        if opts.chunks
        else None
    )
    targets = _targets(client, opts)
    pending: list[tuple] = []  # (video, job_id) for audio jobs
    for start in range(0, len(targets), BATCH_SIZE):
        group = targets[start : start + BATCH_SIZE]
        try:
            resp = client.transcripts.batch(group, mode=opts.mode, chunks=chunk_opts)
        except TranscriptFetchError as e:
            if not opts.continue_on_failure:
                raise
            for video in group:
                _fail(opts, video, str(e))
            continue
        for video, result in zip(group, resp.results):
            if result.outcome == "ok":
                yield from _records(result, video, opts.chunks)
            elif result.outcome == "processing" and result.job_id:
                if opts.wait_for_audio:
                    pending.append((video, result.job_id))
                else:
                    _fail(opts, video, "needs audio transcription (wait_for_audio=False)")
            else:
                err = result.error
                _fail(opts, video, f"{err.code}: {err.message}" if err else "no transcript")
    for video, job_id in pending:
        try:
            t = _await_job(client, opts, video, job_id)
        except TranscriptFetchError as e:
            _fail(opts, video, str(e))
            continue
        if t is not None and t.text:
            yield from _records(t, video, opts.chunks)
        elif t is not None:
            _fail(opts, video, "audio transcription returned no text")
