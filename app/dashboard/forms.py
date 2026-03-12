from django import forms
from .models import Program


class ProgramForm(forms.ModelForm):
    class Meta:
        model = Program
        fields = [
            'name', 'description', 'metric', 'metric_direction',
            'time_budget', 'prepare_cmd', 'train_cmd',
            'editable_files', 'readonly_files',
        ]
        widgets = {
            'name': forms.TextInput(attrs={'placeholder': 'e.g. mnist'}),
            'description': forms.TextInput(attrs={'placeholder': 'Short description'}),
            'metric': forms.TextInput(attrs={'placeholder': 'e.g. val_loss'}),
            'time_budget': forms.NumberInput(attrs={'placeholder': '60'}),
            'prepare_cmd': forms.TextInput(attrs={'placeholder': 'uv run python prepare.py'}),
            'train_cmd': forms.TextInput(attrs={'placeholder': 'uv run python train.py'}),
        }

    editable_files = forms.CharField(
        initial='train.py',
        help_text='Comma-separated list of files the agent can edit',
        widget=forms.TextInput(attrs={'placeholder': 'train.py'}),
    )
    readonly_files = forms.CharField(
        initial='prepare.py',
        required=False,
        help_text='Comma-separated list of read-only files',
        widget=forms.TextInput(attrs={'placeholder': 'prepare.py'}),
    )

    def clean_editable_files(self):
        val = self.cleaned_data['editable_files']
        return [f.strip() for f in val.split(',') if f.strip()]

    def clean_readonly_files(self):
        val = self.cleaned_data['readonly_files']
        if not val:
            return []
        return [f.strip() for f in val.split(',') if f.strip()]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.instance and self.instance.pk:
            self.fields['editable_files'].initial = ', '.join(self.instance.editable_files)
            self.fields['readonly_files'].initial = ', '.join(self.instance.readonly_files)
