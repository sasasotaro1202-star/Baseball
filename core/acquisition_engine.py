from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


@dataclass(frozen=True)
class AcquisitionJob:
    job_id: str
    source: str
    shard: str
    output_path: str
    cache_key: str
    required: bool = True


@dataclass
class AcquisitionResult:
    job_id: str
    source: str
    shard: str
    status: str
    attempts: int
    elapsed_sec: float
    bytes_written: int
    cache_hit: bool
    error: str = ''
    source_timestamp: str = ''


class AcquisitionEngine:
    """Full-scope acquisition executor.

    The engine never shrinks the requested job set when a source is slow or fails.
    It retries failed shards, persists successful checkpoints, and resumes only the
    unfinished shards on the next invocation.

    fetcher(job) must return a dict with:
      content: bytes | str
      source_timestamp: ISO-8601 availability/publication timestamp
      metadata: optional dict
    """

    def __init__(self, cache_dir: Path, checkpoint_path: Path,
                 workers: int = 8, retries: int = 5,
                 backoff_base_sec: float = 1.5):
        self.cache_dir = Path(cache_dir)
        self.checkpoint_path = Path(checkpoint_path)
        self.workers = max(1, int(workers))
        self.retries = max(1, int(retries))
        self.backoff_base_sec = float(backoff_base_sec)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_checkpoint(self) -> dict:
        if not self.checkpoint_path.exists():
            return {'version': 1, 'completed': {}, 'failed': {}}
        return json.loads(self.checkpoint_path.read_text(encoding='utf-8'))

    def _save_checkpoint(self, checkpoint: dict) -> None:
        tmp = self.checkpoint_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, self.checkpoint_path)

    def _cache_path(self, job: AcquisitionJob) -> Path:
        digest = hashlib.sha256(job.cache_key.encode('utf-8')).hexdigest()[:24]
        return self.cache_dir / f'{job.source}_{digest}.cache'

    def _write_atomic(self, path: Path, content: bytes) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix='.partial-', dir=str(path.parent))
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        return len(content)

    def _one(self, job: AcquisitionJob, fetcher: Callable[[AcquisitionJob], dict]) -> AcquisitionResult:
        started = time.monotonic()
        cache = self._cache_path(job)
        if cache.exists() and cache.stat().st_size > 0:
            return AcquisitionResult(
                job.job_id, job.source, job.shard, 'CACHE_HIT', 0,
                time.monotonic() - started, cache.stat().st_size, True
            )

        last_error = ''
        for attempt in range(1, self.retries + 1):
            try:
                payload = fetcher(job)
                content = payload.get('content')
                source_timestamp = str(payload.get('source_timestamp', ''))
                if not source_timestamp:
                    raise ValueError('missing source_timestamp: source availability must be proven')
                if isinstance(content, str):
                    content = content.encode('utf-8')
                if not isinstance(content, (bytes, bytearray)) or not content:
                    raise ValueError('fetcher returned empty/non-byte content')
                size = self._write_atomic(cache, bytes(content))
                return AcquisitionResult(
                    job.job_id, job.source, job.shard, 'FETCHED', attempt,
                    time.monotonic() - started, size, False, '', source_timestamp
                )
            except Exception as exc:
                last_error = repr(exc)
                if attempt < self.retries:
                    delay = self.backoff_base_sec * (2 ** (attempt - 1))
                    delay *= 0.8 + random.random() * 0.4
                    time.sleep(delay)
        return AcquisitionResult(
            job.job_id, job.source, job.shard, 'FAILED', self.retries,
            time.monotonic() - started, 0, False, last_error
        )

    def run(self, jobs: Iterable[AcquisitionJob], fetcher: Callable[[AcquisitionJob], dict]) -> list[AcquisitionResult]:
        jobs = list(jobs)
        checkpoint = self._load_checkpoint()
        completed = checkpoint.setdefault('completed', {})
        pending = [j for j in jobs if j.job_id not in completed]

        results: list[AcquisitionResult] = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self._one, job, fetcher): job for job in pending}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.status in {'FETCHED', 'CACHE_HIT'}:
                    completed[result.job_id] = asdict(result)
                else:
                    checkpoint.setdefault('failed', {})[result.job_id] = asdict(result)
                self._save_checkpoint(checkpoint)

        # A required job may never disappear silently. The caller must decide whether
        # the complete acquisition is allowed to proceed.
        required_failures = [
            r for r in results
            if r.status == 'FAILED' and next(j for j in jobs if j.job_id == r.job_id).required
        ]
        if required_failures:
            raise RuntimeError(f'{len(required_failures)} required acquisition shards failed')
        return results
