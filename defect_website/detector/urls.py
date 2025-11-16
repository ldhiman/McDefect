from django.urls import path
from . import views

urlpatterns = [
    # This maps the root URL ('/') to your 'index' view
    path('', views.index, name='index'),
]