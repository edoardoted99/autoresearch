"""
Scan programs/ directory for autoresearch.toml files and sync to DB.
The toml files are the source of truth — the DB is a cache.
"""
import tomllib
from pathlib import Path
from django.conf import settings


def get_programs_root() -> Path:
    return settings.AUTORESEARCH_REPO_PATH / "programs"


def parse_toml(toml_path: Path) -> dict:
    """Parse an autoresearch.toml into a flat dict matching the Program model fields."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    project = data.get("project", {})
    task = data.get("task", {})
    files = data.get("files", {})
    results = data.get("results", {})
    return {
        "name": project.get("name", toml_path.parent.name),
        "description": project.get("description", ""),
        "metric": task.get("metric", "val_loss"),
        "metric_direction": task.get("metric_direction", "min"),
        "time_budget": task.get("time_budget", 60),
        "prepare_cmd": task.get("prepare_cmd", "uv run python prepare.py"),
        "train_cmd": task.get("train_cmd", "uv run python train.py"),
        "editable_files": files.get("editable", ["train.py"]),
        "readonly_files": files.get("readonly", ["prepare.py"]),
        "results_file": results.get("file", "results.tsv"),
        "results_columns": results.get("columns", []),
    }


def discover_programs() -> list[dict]:
    """Scan programs/ for all autoresearch.toml files and return parsed configs."""
    root = get_programs_root()
    if not root.exists():
        return []
    programs = []
    for toml_path in sorted(root.glob("*/autoresearch.toml")):
        programs.append(parse_toml(toml_path))
    return programs


def sync_programs_to_db():
    """
    Sync discovered toml configs to the DB.
    Creates new programs, updates existing ones, removes stale ones.
    """
    from .models import Program

    discovered = {p["name"]: p for p in discover_programs()}

    # Update or create
    for name, config in discovered.items():
        Program.objects.update_or_create(
            name=name,
            defaults={
                "description": config["description"],
                "metric": config["metric"],
                "metric_direction": config["metric_direction"],
                "time_budget": config["time_budget"],
                "prepare_cmd": config["prepare_cmd"],
                "train_cmd": config["train_cmd"],
                "editable_files": config["editable_files"],
                "readonly_files": config["readonly_files"],
            },
        )

    # Remove programs whose toml no longer exists
    Program.objects.exclude(name__in=discovered.keys()).delete()


def write_toml(program) -> Path:
    """Write autoresearch.toml from a Program model instance. Returns the path."""
    root = get_programs_root()
    program_dir = root / program.name
    program_dir.mkdir(parents=True, exist_ok=True)
    toml_path = program_dir / "autoresearch.toml"

    editable = ", ".join(f'"{f}"' for f in program.editable_files)
    readonly = ", ".join(f'"{f}"' for f in program.readonly_files)

    content = f"""[project]
name = "{program.name}"
description = "{program.description}"

[task]
time_budget = {program.time_budget}
metric = "{program.metric}"
metric_direction = "{program.metric_direction}"
prepare_cmd = "{program.prepare_cmd}"
train_cmd = "{program.train_cmd}"

[files]
editable = [{editable}]
readonly = [{readonly}]
agent_instructions = "program.md"

[results]
file = "results.tsv"
columns = ["commit", "{program.metric}", "memory_gb", "status", "description"]
"""
    toml_path.write_text(content)
    return toml_path
