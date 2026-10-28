"""LlamaIndex reader for TranscriptFetch."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from typing import Literal

from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

from transcriptfetch import TranscriptFetch

from ._core import FetchOptions, iter_records, normalize_videos

# Kept on the node for citations and filtering, left out of what gets embedded.
_NOT_EMBEDDED = [
    "source", "video_url", "video_id", "chunk_index", "start", "end", "tokens",
    "duration", "language", "transcript_source",
]


class TranscriptFetchReader(BasePydanticReader):
    """Transcripts of YouTube, TikTok and Instagram videos as Documents.

    Configure the reader once, then call :meth:`load_data` with exactly one
    source: ``videos`` (a URL or ID, or a list), ``channel``, ``playlist`` or
    ``search``. With ``chunks=True`` the API splits each transcript into pieces
    sized for an embedding model, one Document per piece, with ``start`` /
    ``end`` seconds and a ``source`` link to that moment.

    Example:
        .. code-block:: python

            from llama_index.readers.transcriptfetch import TranscriptFetchReader

            reader = TranscriptFetchReader(chunks=True)  # TRANSCRIPTFETCH_API_KEY from env
            docs = reader.load_data(channel="@3blue1brown", max_videos=10)

    Args:
        api_key: TranscriptFetch API key; defaults to ``TRANSCRIPTFETCH_API_KEY``.
        mode: ``"auto"`` (default) transcribes the audio of videos without
            captions; ``"captions"`` never does, so those are skipped.
        chunks: One Document per embedding-sized piece instead of per video.
        chunk_tokens: Target chunk size in tokens (100 to 2000).
        chunk_overlap: Tokens shared with the previous chunk (0 to 200).
        wait_for_audio: Wait for audio transcriptions to finish (default).
        audio_timeout: Seconds to wait for one audio transcription.
        continue_on_failure: Log and skip a failing video instead of raising.
        base_url: API base URL override.
    """

    is_remote: bool = True
    api_key: str | None = None
    mode: Literal["auto", "captions"] = "auto"
    chunks: bool = False
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    wait_for_audio: bool = True
    audio_timeout: float = 900.0
    continue_on_failure: bool = True
    base_url: str | None = None

    @classmethod
    def class_name(cls) -> str:
        return "TranscriptFetchReader"

    def _client(self) -> TranscriptFetch:
        kwargs = {"api_key": self.api_key or os.environ.get("TRANSCRIPTFETCH_API_KEY")}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return TranscriptFetch(**kwargs)

    def lazy_load_data(
        self,
        videos: str | Iterable[str] | None = None,
        *,
        channel: str | None = None,
        playlist: str | None = None,
        search: str | None = None,
        search_platform: Literal["youtube", "tiktok", "instagram"] | None = None,
        max_videos: int = 25,
    ) -> Iterator[Document]:
        """Yield one Document per video, or per chunk with ``chunks=True``.

        Args:
            videos: A video URL or ID, or a list of them (YouTube, TikTok,
                Instagram, or a direct audio or video file URL).
            channel: A YouTube channel (@handle, URL or ID), or a TikTok or
                Instagram profile URL.
            playlist: A YouTube or TikTok playlist URL or ID.
            search: A search query; YouTube unless ``search_platform`` says otherwise.
            search_platform: ``"youtube"``, ``"tiktok"`` or ``"instagram"``.
            max_videos: Most videos to take from a channel, playlist or search.
        """
        opts = FetchOptions(
            videos=normalize_videos(videos),
            channel=channel,
            playlist=playlist,
            search=search,
            search_platform=search_platform,
            max_videos=max_videos,
            mode=self.mode,
            chunks=self.chunks,
            chunk_tokens=self.chunk_tokens,
            chunk_overlap=self.chunk_overlap,
            wait_for_audio=self.wait_for_audio,
            audio_timeout=self.audio_timeout,
            continue_on_failure=self.continue_on_failure,
        )
        with self._client() as client:
            for record in iter_records(client, opts):
                yield Document(
                    text=record.text,
                    metadata=record.metadata,
                    id_=record.id,
                    excluded_embed_metadata_keys=[k for k in _NOT_EMBEDDED if k in record.metadata],
                    excluded_llm_metadata_keys=["video_id", "chunk_index", "tokens"],
                )

    def load_data(self, *args, **kwargs) -> list[Document]:  # type: ignore[override]
        """Load everything at once. Same arguments as :meth:`lazy_load_data`."""
        return list(self.lazy_load_data(*args, **kwargs))
