"""
Reporter module for displaying rich terminal summary tables and exporting execution reports.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class ExecutionReporter:
    """Prints and exports summary reports for PicSort execution."""

    @staticmethod
    def print_summary(
        total_photos: int,
        processed_photos: int,
        cached_photos: int,
        failed_photos: int,
        total_faces: int,
        person_counts: Dict[str, int],
        elapsed_seconds: float,
    ):
        """Display a formatted rich summary report in the terminal."""
        console.print()
        console.print(
            Panel.fit(
                "[bold green]PicSort Processing Summary[/bold green]",
                subtitle=f"Elapsed Time: {elapsed_seconds:.2f}s",
            )
        )

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")

        table.add_row("Total Photos Scanned", str(total_photos))
        table.add_row("Newly Processed", str(processed_photos))
        table.add_row("Loaded from Cache", str(cached_photos))
        table.add_row("Failed / Unreadable", str(failed_photos))
        table.add_row("Total Faces Detected", str(total_faces))
        table.add_row("Total Categories", str(len(person_counts)))

        console.print(table)
        console.print()

        # Breakdown table per person
        person_table = Table(
            show_header=True, header_style="bold magenta", title="Photos per Person / Category"
        )
        person_table.add_column("Category / Person Name", style="bold")
        person_table.add_column("Photo Count", justify="right")

        for person, count in sorted(person_counts.items(), key=lambda x: x[1], reverse=True):
            person_table.add_row(person, str(count))

        console.print(person_table)
        console.print()

    @staticmethod
    def save_report_json(
        output_path: Path,
        summary_data: Dict[str, Any],
    ):
        """Export execution summary data to a JSON file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            console.print(f"[dim]Saved report summary to '{output_path}'[/dim]")
        except Exception as e:
            console.print(f"[bold red]Failed to save JSON report: {e}[/bold red]")
