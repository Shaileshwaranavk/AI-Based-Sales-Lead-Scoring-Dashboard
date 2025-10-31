from django.urls import path
from .views import analyze_leads

urlpatterns = [
    path("analyze/", analyze_leads, name="analyze_leads"),
]
