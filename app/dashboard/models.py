from django.db import models


class Program(models.Model):
    name = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    # task config
    metric = models.CharField(max_length=50, default="val_loss")
    metric_direction = models.CharField(
        max_length=3,
        choices=[("min", "Lower is better"), ("max", "Higher is better")],
        default="min",
    )
    time_budget = models.IntegerField(default=60, help_text="Training time budget in seconds")
    prepare_cmd = models.CharField(max_length=500, default="uv run python prepare.py")
    train_cmd = models.CharField(max_length=500, default="uv run python train.py")
    # files config
    editable_files = models.JSONField(default=list, help_text='e.g. ["train.py"]')
    readonly_files = models.JSONField(default=list, help_text='e.g. ["prepare.py"]')
    # meta
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return self.name

    @property
    def metric_is_lower_better(self):
        return self.metric_direction == "min"

    @property
    def branch_prefix(self):
        return f"autoresearch/{self.name}"

    @property
    def programs_dir(self):
        """Path to this program's files in the repo."""
        from django.conf import settings
        return settings.AUTORESEARCH_REPO_PATH / "programs" / self.name

    @property
    def program_md_path(self):
        return self.programs_dir / "program.md"

    @property
    def all_files(self):
        return self.editable_files + self.readonly_files
