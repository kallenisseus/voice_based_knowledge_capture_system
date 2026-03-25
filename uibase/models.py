from django.db import models
from django.utils import timezone


class AudioUpload(models.Model):
    stored_name = models.CharField(max_length=255, unique=True)
    original_name = models.CharField(max_length=255, blank=True)
    category = models.CharField(max_length=255, db_index=True)
    author = models.CharField(max_length=255, blank=True)
    recorded_on = models.DateField(default=timezone.localdate)
    transcription_text = models.TextField(blank=True)
    language = models.CharField(max_length=32, blank=True)
    duration = models.FloatField(default=0.0)
    processing_error = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self) -> str:
        return self.stored_name


class TranscriptSegment(models.Model):
    upload = models.ForeignKey(
        AudioUpload,
        related_name="segments",
        on_delete=models.CASCADE,
    )
    position = models.PositiveIntegerField()
    start = models.FloatField(null=True, blank=True)
    end = models.FloatField(null=True, blank=True)
    text = models.TextField()
    embedding = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["position", "id"]
        constraints = [
            models.UniqueConstraint(
                fields=["upload", "position"],
                name="unique_segment_position_per_upload",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.upload.stored_name}#{self.position}"


class CategoryCluster(models.Model):
    category = models.CharField(max_length=255, db_index=True)
    name = models.CharField(max_length=255)
    summary = models.TextField(blank=True)
    summary_sections = models.JSONField(default=list, blank=True)
    embedding = models.JSONField(default=list, blank=True)
    member_count = models.PositiveIntegerField(default=0)
    position = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["category", "position", "id"]

    def __str__(self) -> str:
        return f"{self.category}: {self.name}"


class ClusterSegment(models.Model):
    cluster = models.ForeignKey(
        CategoryCluster,
        related_name="memberships",
        on_delete=models.CASCADE,
    )
    segment = models.ForeignKey(
        TranscriptSegment,
        related_name="cluster_memberships",
        on_delete=models.CASCADE,
    )
    position = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["position", "id"]
        constraints = [
            models.UniqueConstraint(
                fields=["cluster", "segment"],
                name="unique_segment_per_cluster",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.cluster} -> {self.segment}"
