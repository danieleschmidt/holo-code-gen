"""Command-line interface for Holo-Code-Gen."""

import typer
from rich.console import Console
from rich.table import Table

from . import __version__

app = typer.Typer(
    name="holo-code-gen",
    help="HLS toolchain for photonic neural networks",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"Holo-Code-Gen version {__version__}")


@app.command()
def compile(
    model_path: str = typer.Argument(..., help="Path to neural network model"),
    output_dir: str = typer.Option("./output", help="Output directory"),
    target: str = typer.Option("imec", help="Target photonic platform"),
):
    """Compile neural network to photonic circuit."""
    console.print(f"Compiling {model_path} for {target} platform...")
    console.print(f"Output will be saved to {output_dir}")
    # TODO: Implement compilation logic
    console.print("[red]Not yet implemented[/red]")


@app.command()
def simulate(
    circuit_path: str = typer.Argument(..., help="Path to photonic circuit"),
    method: str = typer.Option("fdtd", help="Simulation method"),
):
    """Simulate photonic circuit performance."""
    console.print(f"Simulating {circuit_path} using {method} method...")
    # TODO: Implement simulation logic
    console.print("[red]Not yet implemented[/red]")


@app.command()
def list_templates():
    """List available photonic component templates."""
    table = Table(title="Available Photonic Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")
    
    # TODO: Load actual templates
    table.add_row("microring_weight", "Weight Bank", "Microring resonator weight bank")
    table.add_row("mzi_mesh", "Matrix", "Mach-Zehnder interferometer mesh")
    table.add_row("ring_modulator", "Nonlinearity", "Ring modulator for activation")
    
    console.print(table)


def main():
    """Main entry point."""
    app()