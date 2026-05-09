from django.urls import path
from . import views

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("uploads/<int:upload_id>/delete/", views.delete_upload, name="delete_upload"),
    path("uploads/delete-many/", views.bulk_delete_uploads, name="bulk_delete_uploads"),
    path("summaries/redo-category/", views.redo_category_summary, name="redo_category_summary"),
    path("summaries/redo-cluster/<int:cluster_id>/", views.redo_cluster_summary_view, name="redo_cluster_summary"),
    path("categories/rename/", views.rename_category_view, name="rename_category"),
    path("categories/set-color/", views.set_machine_type_color_view, name="set_machine_type_color"),
    path("categories/remove/", views.remove_category, name="remove_category"),
    path("clusters/<int:cluster_id>/rename/", views.rename_cluster_view, name="rename_cluster"),
    path("clusters/<int:cluster_id>/merge/", views.merge_cluster_view, name="merge_cluster"),
    path("clusters/<int:cluster_id>/remove/", views.remove_cluster_view, name="remove_cluster"),
    path("machines/set-color/", views.set_machine_color_view, name="set_machine_color"),
    path("segments/<int:segment_id>/move/", views.move_segment_view, name="move_segment"),
    path("segments/<int:segment_id>/delete/", views.delete_segment_view, name="delete_segment"),
]
