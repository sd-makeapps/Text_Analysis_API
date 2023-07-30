from django.urls import path, include
from analysis_api.api.views import AnalysisView

urlpatterns = [
    path("", AnalysisView.as_view({'post': 'analysis'}), name='analysis'),
]
