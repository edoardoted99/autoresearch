from django.urls import path
from . import views

app_name = 'dashboard'
urlpatterns = [
    # Home
    path('', views.index, name='index'),
    # Programs CRUD
    path('programs/new/', views.program_create, name='program_create'),
    path('programs/<slug:program_name>/', views.program_detail, name='program_detail'),
    path('programs/<slug:program_name>/edit/', views.program_edit, name='program_edit'),
    path('programs/<slug:program_name>/program-md/', views.program_md_edit, name='program_md_edit'),
    # Branches & Experiments
    path('programs/<slug:program_name>/branch/<path:branch_tag>/', views.branch_detail, name='branch_detail'),
    path('programs/<slug:program_name>/commit/<str:commit_hash>/', views.commit_detail, name='commit_detail'),
    # GPU
    path('gpu/', views.gpu_stats, name='gpu_stats'),
    path('api/gpu/', views.gpu_stats_api, name='gpu_stats_api'),
    # Experiment status
    path('api/experiments/status/', views.experiment_status_api, name='experiment_status_api'),
]
