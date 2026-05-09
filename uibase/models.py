from django.db import models
from django.utils import timezone


class AudioUpload(models.Model):
    stored_name = models.CharField(max_length=255, unique=True)
    original_name = models.CharField(max_length=255, blank=True)
    machine_name = models.CharField(max_length=255, db_index=True, blank=True)
    machine_type = models.CharField(max_length=255, db_index=True, default="Unassigned")
    # Each item is one optional taxonomy path, e.g.
    # [["Brakes", "Brake pads"], ["Wheels", "Torque"]]
    subcategory_paths = models.JSONField(default=list, blank=True)
    hierarchy_path = models.JSONField(default=list, blank=True)
    extra_tags = models.JSONField(default=list, blank=True)
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
    needs_resummary = models.BooleanField(default=False)
    stale_deleted_count = models.PositiveIntegerField(default=0)
    stale_deleted_files = models.JSONField(default=list, blank=True)
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


class MachineTypeStyle(models.Model):
    machine_type = models.CharField(max_length=255, unique=True)
    color_hex = models.CharField(max_length=7, default="#3A78F2")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["machine_type"]

    def __str__(self) -> str:
        return f"{self.machine_type}: {self.color_hex}"


class MachineStyle(models.Model):
    machine_name = models.CharField(max_length=255, unique=True)
    color_hex = models.CharField(max_length=7, default="#3A78F2")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["machine_name"]

    def __str__(self) -> str:
        return f"{self.machine_name}: {self.color_hex}"
