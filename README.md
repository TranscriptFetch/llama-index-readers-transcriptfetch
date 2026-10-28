# LlamaIndex Readers Integration: TranscriptFetch

Transcripts of YouTube, TikTok and Instagram videos as LlamaIndex Documents, from a single link to whole channels, playlists and searches, via [TranscriptFetch](https://transcriptfetch.com). With `chunks=True` the API splits each transcript into embedding-sized pieces with a link to the moment each one starts, so answers can cite the exact second.

```bash
pip install llama-index-readers-transcriptfetch
export TRANSCRIPTFETCH_API_KEY="tf_live_..."   # free key: https://transcriptfetch.com
```

## Usage

```python
from llama_index.readers.transcriptfetch import TranscriptFetchReader

reader = TranscriptFetchReader()
docs = reader.load_data(["https://www.youtube.com/watch?v=aircAruvnKk"])
```

A whole channel, chunked, into an index:

```python
from llama_index.core import VectorStoreIndex

reader = TranscriptFetchReader(chunks=True)
docs = reader.load_data(channel="@3blue1brown", max_videos=20)

index = VectorStoreIndex.from_documents(docs)
response = index.as_query_engine().query("What does a neuron hold?")
for node in response.source_nodes:
    print(node.metadata["source"])  # https://www.youtube.com/watch?v=...&t=4s
```

Each chunk Document carries `source` (a link to that moment), `video_url`, `chunk_index`, `start`, `end`, `tokens`, `title`, `channel` and `platform`. Links, IDs and timings are kept out of the embedded text.

## Sources

`load_data()` takes exactly one: `videos` (a URL or ID, or a list: YouTube, TikTok, Instagram, or a direct audio or video file URL), `channel` (a YouTube `@handle`, URL or ID, or a TikTok or Instagram profile URL), `playlist`, or `search` (YouTube unless `search_platform="tiktok"` or `"instagram"`). Listings are capped at `max_videos` (default 25). `lazy_load_data()` streams instead.

## Options

Set on the reader: `mode` (`"auto"` transcribes the audio of videos without captions; `"captions"` skips them and costs nothing), `chunks`, `chunk_tokens` (100 to 2000, default 500), `chunk_overlap` (0 to 200, default 50), `wait_for_audio`, `audio_timeout`, `continue_on_failure`, `api_key`, `base_url`.

## Billing

A transcript or a listing page costs 1 credit, audio transcription 1 credit per started 5 minutes of audio, and failed requests are free. Every account gets 50 free credits a month, no card needed.
