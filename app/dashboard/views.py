import json
import subprocess
import markdown
from datetime import datetime, timezone
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.conf import settings
from django.utils.safestring import mark_safe
from . import git_service
from .models import Program
from .forms import ProgramForm
from .protocol import write_toml, sync_programs_to_db


# --- Programs ---

def index(request):
    """Home — list all programs."""
    sync_programs_to_db()  # toml is source of truth, sync on every page load
    programs = Program.objects.all()
    return render(request, 'dashboard/index.html', {'programs': programs})


def program_create(request):
    if request.method == 'POST':
        form = ProgramForm(request.POST)
        if form.is_valid():
            program = form.save()
            # write toml + scaffold program.md
            write_toml(program)
            if not program.program_md_path.exists():
                program.program_md_path.write_text(f"# {program.name}\n\nInstructions for the agent.\n")
            return redirect('dashboard:program_detail', program_name=program.name)
    else:
        form = ProgramForm()
    return render(request, 'dashboard/program_form.html', {'form': form, 'title': 'New Program'})


def program_edit(request, program_name):
    program = get_object_or_404(Program, name=program_name)
    if request.method == 'POST':
        form = ProgramForm(request.POST, instance=program)
        if form.is_valid():
            program = form.save()
            write_toml(program)  # keep toml in sync
            return redirect('dashboard:program_detail', program_name=program.name)
    else:
        form = ProgramForm(instance=program)
    return render(request, 'dashboard/program_form.html', {'form': form, 'title': f'Edit {program.name}'})


def program_detail(request, program_name):
    """Program overview — config + branches."""
    program = get_object_or_404(Program, name=program_name)
    branches = git_service.get_branches_for_program(program)
    # read program.md and render as HTML
    program_md_html = ''
    if program.program_md_path.exists():
        raw = program.program_md_path.read_text()
        program_md_html = mark_safe(markdown.markdown(
            raw, extensions=['fenced_code', 'tables', 'codehilite']
        ))
    return render(request, 'dashboard/program_detail.html', {
        'program': program,
        'branches': branches,
        'program_md_html': program_md_html,
    })


def program_md_edit(request, program_name):
    """Edit program.md via the UI."""
    program = get_object_or_404(Program, name=program_name)
    program.programs_dir.mkdir(parents=True, exist_ok=True)
    if request.method == 'POST':
        content = request.POST.get('content', '')
        program.program_md_path.write_text(content)
        return redirect('dashboard:program_detail', program_name=program.name)
    content = ''
    if program.program_md_path.exists():
        content = program.program_md_path.read_text()
    return render(request, 'dashboard/program_md_edit.html', {
        'program': program,
        'content': content,
    })


# --- Branches / Experiments / Commits ---

def branch_detail(request, program_name, branch_tag):
    """Experiments table + chart for a branch."""
    program = get_object_or_404(Program, name=program_name)
    full_name = f'{program.branch_prefix}/{branch_tag}'
    experiments = git_service.get_experiments_for_branch(program, full_name)
    commits = git_service.get_branch_commits(full_name)

    # compute stats
    lower = program.metric_is_lower_better
    valid = [e for e in experiments if e.status == 'keep' and e.val_metric > 0]
    best_metric = None
    if valid:
        best_metric = min(e.val_metric for e in valid) if lower else max(e.val_metric for e in valid)
    # mark best experiment and add is_best flag
    for e in experiments:
        e.is_best = (e.val_metric == best_metric and e.val_metric > 0) if best_metric else False
    keep_count = sum(1 for e in experiments if e.status == 'keep')
    discard_count = sum(1 for e in experiments if e.status == 'discard')
    crash_count = sum(1 for e in experiments if e.status == 'crash')

    chart_data = json.dumps([
        {'x': i + 1, 'val': e.val_metric, 'label': e.description, 'is_best': e.is_best}
        for i, e in enumerate(experiments)
        if e.val_metric > 0
    ])
    return render(request, 'dashboard/branch_detail.html', {
        'program': program,
        'branch_tag': branch_tag,
        'full_name': full_name,
        'experiments': experiments,
        'commits': commits,
        'chart_data': chart_data,
        'has_chart': len([e for e in experiments if e.val_metric > 0]) > 0,
        'best_metric': best_metric,
        'keep_count': keep_count,
        'discard_count': discard_count,
        'crash_count': crash_count,
    })


def commit_detail(request, program_name, commit_hash):
    """Show diff for a specific commit."""
    program = get_object_or_404(Program, name=program_name)
    info = git_service.get_commit_diff(commit_hash, program.editable_files)
    return render(request, 'dashboard/commit_detail.html', {
        'program': program,
        'commit': info,
    })


# --- GPU ---

def experiment_status_api(request):
    """JSON endpoint: running experiments across all programs."""
    programs = Program.objects.all()
    statuses = []
    now = datetime.now(timezone.utc)
    repo_root = settings.AUTORESEARCH_REPO_PATH
    for program in programs:
        # check programs/<name>/status.json first, then repo root
        status_path = program.programs_dir / "status.json"
        if not status_path.exists():
            status_path = repo_root / "status.json"
        if not status_path.exists():
            continue
        try:
            data = json.loads(status_path.read_text())
            if not data.get("running"):
                continue
            started = datetime.fromisoformat(data["started_at"])
            elapsed = (now - started).total_seconds()
            budget = data.get("time_budget", 60)
            # stale detection: if elapsed > 2x budget, clean up
            if elapsed > budget * 2:
                status_path.unlink(missing_ok=True)
                continue
            statuses.append({
                "program": program.name,
                "elapsed": round(elapsed, 1),
                "time_budget": budget,
                "progress": min(100, round(elapsed / budget * 100, 1)),
            })
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return JsonResponse({"experiments": statuses})


def gpu_stats(request):
    return render(request, 'dashboard/gpu_stats.html')


def gpu_stats_api(request):
    """JSON endpoint for GPU stats."""
    try:
        result = subprocess.run(
            ['nvidia-smi',
             '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return JsonResponse({'error': result.stderr.strip()}, status=500)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip():
                continue
            p = [x.strip() for x in line.split(',')]
            gpus.append({
                'index': int(p[0]), 'name': p[1],
                'gpu_util': float(p[2]) if p[2] != 'N/A' else 0,
                'mem_util': float(p[3]) if p[3] != 'N/A' else 0,
                'mem_used': float(p[4]) if p[4] != 'N/A' else 0,
                'mem_total': float(p[5]) if p[5] != 'N/A' else 0,
                'temperature': float(p[6]) if p[6] != 'N/A' else 0,
                'power_draw': float(p[7]) if p[7] != 'N/A' else 0,
                'power_limit': float(p[8]) if p[8] != 'N/A' else 0,
            })
        proc_result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory,process_name',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        processes = []
        if proc_result.returncode == 0:
            for line in proc_result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                p = [x.strip() for x in line.split(',')]
                if len(p) >= 3:
                    processes.append({'pid': p[0], 'mem_used': p[1], 'name': p[2]})
        return JsonResponse({'gpus': gpus, 'processes': processes})
    except FileNotFoundError:
        return JsonResponse({'error': 'nvidia-smi not found'}, status=404)
    except subprocess.TimeoutExpired:
        return JsonResponse({'error': 'nvidia-smi timeout'}, status=500)
