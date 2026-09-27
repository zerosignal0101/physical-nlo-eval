"""CLI entry point for physics-nlo-eval."""

import json
from dataclasses import asdict
from pathlib import Path

import click

from physics_nlo_eval.channel.assessment import FiberSpan, NLIChannel, assess_fiber_nonlinearity


@click.group()
@click.version_option()
def main() -> None:
    """Physics NLO Eval - Optical Network Physical Layer Simulation."""
    pass


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
def simulate(config: Path) -> None:
    """Run a fast analytical route assessment from a JSON config."""
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        channels = [NLIChannel(**item) for item in payload["channels"]]
        spans = [FiberSpan(**item) for item in payload["spans"]]
        results = assess_fiber_nonlinearity(
            channels, spans, coherence_exponent=float(payload.get("coherence_exponent", 0.0))
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(f"Invalid assessment config: {exc}") from exc
    click.echo(json.dumps([asdict(result) for result in results], indent=2, allow_nan=True))


@main.command("assess")
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
def assess(config: Path) -> None:
    """Alias for the analytical simulation command."""
    simulate.callback(config=config)


if __name__ == "__main__":
    main()
