"""Service layer to read experiment data from the git repo."""
import csv
import io
from dataclasses import dataclass
from git import Repo
from django.conf import settings


def get_repo():
    return Repo(settings.AUTORESEARCH_REPO_PATH)


@dataclass
class Experiment:
    commit: str
    val_metric: float
    metric_name: str
    memory_gb: float
    status: str
    description: str


@dataclass
class BranchInfo:
    name: str
    tag: str  # the part after autoresearch/<program>/
    num_experiments: int
    best_metric: float | None
    last_commit_message: str
    last_commit_date: str


def get_branches_for_program(program):
    """List all branches for a program, deduped (local preferred)."""
    repo = get_repo()
    prefix = program.branch_prefix + "/"  # e.g. "autoresearch/mnist/"
    seen = {}
    for ref in repo.refs:
        name = ref.name
        # match both local and remote
        if prefix in name:
            tag = name.split(prefix)[-1]
            if not tag:
                continue
            is_local = not name.startswith('origin/')
            if tag in seen and not is_local:
                continue
            seen[tag] = (name, ref)
    branches = []
    lower_better = program.metric_is_lower_better
    for tag, (name, ref) in seen.items():
        experiments = get_experiments_for_branch(program, name)
        best = None
        if experiments:
            valid = [e for e in experiments if e.status == 'keep' and e.val_metric > 0]
            if valid:
                best = min(e.val_metric for e in valid) if lower_better else max(e.val_metric for e in valid)
        commit = ref.commit
        branches.append(BranchInfo(
            name=name,
            tag=tag,
            num_experiments=len(experiments),
            best_metric=best,
            last_commit_message=commit.message.strip().split('\n')[0],
            last_commit_date=commit.committed_datetime.strftime('%Y-%m-%d %H:%M'),
        ))
    return branches


def get_experiments_for_branch(program, branch_name):
    """Read results.tsv from a given branch (git tree first, then disk fallback)."""
    repo = get_repo()
    content = None
    # Try git tree first
    try:
        ref_names = [r.name for r in repo.refs]
        commit = repo.refs[branch_name].commit if branch_name in ref_names else repo.commit(branch_name)
        for path in [f'programs/{program.name}/results.tsv', 'results.tsv']:
            try:
                blob = commit.tree / path
                content = blob.data_stream.read().decode('utf-8')
                break
            except (KeyError, TypeError):
                continue
    except (KeyError, TypeError):
        pass
    # Fallback: read from disk if on the active branch
    if content is None:
        from django.conf import settings
        for path in [
            settings.AUTORESEARCH_REPO_PATH / 'programs' / program.name / 'results.tsv',
            settings.AUTORESEARCH_REPO_PATH / 'results.tsv',
        ]:
            if path.exists():
                content = path.read_text()
                break
    if content is None:
        return []
    experiments = []
    reader = csv.DictReader(io.StringIO(content), delimiter='\t')
    metric_col = program.metric
    # fallback: detect from header
    if reader.fieldnames and metric_col not in reader.fieldnames:
        for col in reader.fieldnames:
            if col.startswith('val_'):
                metric_col = col
                break
    for row in reader:
        try:
            experiments.append(Experiment(
                commit=row.get('commit', ''),
                val_metric=float(row.get(metric_col, 0)),
                metric_name=metric_col,
                memory_gb=float(row.get('memory_gb', 0)),
                status=row.get('status', ''),
                description=row.get('description', ''),
            ))
        except (ValueError, KeyError):
            continue
    return experiments


def get_commit_diff(commit_hash, editable_files=None):
    """Get the diff for a specific commit."""
    repo = get_repo()
    editable_files = editable_files or []
    commit = repo.commit(commit_hash)
    if commit.parents:
        diff = commit.parents[0].diff(commit, create_patch=True)
    else:
        diff = commit.diff(None, create_patch=True)
    diffs = []
    for d in diff:
        filepath = d.b_path or d.a_path
        diffs.append({
            'file': filepath,
            'is_editable': filepath in editable_files,
            'patch': d.diff.decode('utf-8', errors='replace') if d.diff else '',
        })
    return {
        'hash': commit.hexsha[:7],
        'message': commit.message.strip(),
        'author': str(commit.author),
        'date': commit.committed_datetime.strftime('%Y-%m-%d %H:%M'),
        'diffs': diffs,
    }


def get_branch_commits(branch_name, max_count=50):
    """Get recent commits on a branch."""
    repo = get_repo()
    try:
        ref = next(r for r in repo.refs if r.name == branch_name)
    except StopIteration:
        return []
    commits = []
    for c in repo.iter_commits(ref.commit, max_count=max_count):
        commits.append({
            'hash': c.hexsha[:7],
            'full_hash': c.hexsha,
            'message': c.message.strip().split('\n')[0],
            'date': c.committed_datetime.strftime('%Y-%m-%d %H:%M'),
        })
    return commits
