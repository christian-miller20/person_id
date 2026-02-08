from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .face_embedder import FaceEmbedder
from .face_pipeline import FacePipeline
from .identity_config import IdentityConfig
from .identity_engine import IdentityEngine
from .identity_store import IdentityStore

app = typer.Typer(add_completion=False, help="Face-based identity pipeline.")


def _build_identity(store_path: Path) -> IdentityEngine:
    store = IdentityStore(store_path)
    config = IdentityConfig()
    return IdentityEngine(store=store, config=config)


@app.command()
def identify(
    source: str = typer.Argument(..., help="Path to video file or camera index."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
    limit_frames: Optional[int] = typer.Option(
        None, "--limit-frames", help="Stop after N frames for quick checks."
    ),
) -> None:
    identity = _build_identity(store_path)
    pipeline = FacePipeline(embedder=FaceEmbedder(), identity=identity)
    result = pipeline.process_video(source, limit_frames=limit_frames)
    typer.secho(
        " ".join(
            [
                f"accepted={result.accepted}",
                f"user_id={result.decision_user_id}",
                f"score={result.decision_score:.3f}",
                f"margin={result.decision_margin:.3f}",
                f"n_used={result.n_used}",
                f"dispersion={result.dispersion:.3f}",
                f"reason={result.reason}",
            ]
        ),
        fg=typer.colors.CYAN,
    )


@app.command()
def add_user(
    user_id: str = typer.Argument(..., help="User ID to create in the store."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    store.add_user(user_id)
    typer.secho(f"Added {user_id}", fg=typer.colors.GREEN)


@app.command()
def list_users(
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    for user_id in store.list_users():
        typer.echo(user_id)


@app.command()
def delete_user(
    user_id: str = typer.Argument(..., help="User ID to delete from the store."),
    store_path: Path = typer.Option(
        Path("profiles/face_templates.json"), "--store", help="Template store path."
    ),
) -> None:
    store = IdentityStore(store_path)
    if store.delete_user(user_id):
        typer.secho(f"Deleted {user_id}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"User {user_id} not found", fg=typer.colors.YELLOW)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
