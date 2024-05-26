# your_app_name/urls.py
from django.urls import path
from .views import generate_text

urlpatterns = [
    path('', generate_text, name='generate_text'),
]
