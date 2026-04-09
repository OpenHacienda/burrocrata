"""Push an adapter or merged model to HuggingFace Hub."""

from __future__ import annotations

import logging
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


def run(config: Config, source_dir: Path) -> str:
    """Upload ``source_dir`` to ``config.hub.repo``. Returns the repo id."""
    if not config.hub.push or not config.hub.repo:
        raise RuntimeError("hub.push is false or hub.repo is unset; nothing to push")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(config.hub.repo, exist_ok=True)
    api.upload_folder(
        folder_path=str(source_dir),
        repo_id=config.hub.repo,
        repo_type="model",
    )
    logger.info("Pushed %s → %s", source_dir, config.hub.repo)
    return config.hub.repo
