from django.contrib import admin

from .models import AudioUpload, CategoryCluster, ClusterSegment, TranscriptSegment


class TranscriptSegmentInline(admin.TabularInline):
    model = TranscriptSegment
    extra = 0
    readonly_fields = ("position", "start", "end", "text")


@admin.register(AudioUpload)
class AudioUploadAdmin(admin.ModelAdmin):
    list_display = (
        "stored_name",
        "category",
        "author",
        "recorded_on",
        "duration",
        "created_at",
    )
    list_filter = ("category", "recorded_on", "created_at")
    search_fields = ("stored_name", "original_name", "author", "category")
    inlines = [TranscriptSegmentInline]


class ClusterSegmentInline(admin.TabularInline):
    model = ClusterSegment
    extra = 0


@admin.register(CategoryCluster)
class CategoryClusterAdmin(admin.ModelAdmin):
    list_display = ("category", "name", "member_count", "position", "created_at")
    list_filter = ("category",)
    search_fields = ("category", "name", "summary")
    inlines = [ClusterSegmentInline]
