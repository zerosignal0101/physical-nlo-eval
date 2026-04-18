"""CLI entry point for physics-nlo-eval."""

from pathlib import Path

import click


@click.group()
@click.version_option()
def main() -> None:
    """Physics NLO Eval - Optical Network Physical Layer Simulation."""
    pass


@main.command()
@click.option("--config", type=click.Path(exists=True, path_type=Path), required=True)
def simulate(config: Path) -> None:
    """Run physics simulation with given config."""
    click.echo(f"Simulation not yet implemented. Config: {config}")


if __name__ == "__main__":
    main()
