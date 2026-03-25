from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("uploads/<int:upload_id>/delete/", views.delete_upload, name="delete_upload"),
]
