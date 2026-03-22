from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .utils import read_json_data
import os

def _ensure_audio_dir():
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        return None
    audio_dir = os.path.join(media_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir


def returnindex(request):
    # Handle POST submission for file upload and metadata
    context = {}
    if request.method == "POST":
        uploaded = request.FILES.get("audio_file")
        category = request.POST.get("category", "").strip()
        author_name = request.POST.get("author_name", "").strip()

        print(uploaded, category, author_name)
        saved_name = None
        error = None
        if uploaded:
            target_dir = _ensure_audio_dir()
            if target_dir:
                fs = FileSystemStorage(location=target_dir)
                saved_name = fs.save(uploaded.name, uploaded)
                context.update({
                    "post_success": True,
                    "saved_file": saved_name,
                    "category_submitted": category,
                    "author_submitted": author_name,
                })
            else:
                error = "MEDIA_ROOT is not configured."
        else:
            error = "No file uploaded."

        if error:
            context.update({
                "post_success": False,
                "error": error,
                "category_submitted": category,
                "author_submitted": author_name,
            })

        if uploaded and not error:
            from ai.ai_process import process_file

            process_file(uploaded.name, author_name, category)

    # Load data for display regardless of method
    raw = read_json_data() or {}
    kb = raw.get("data", {}) if isinstance(raw, dict) else {}
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    context.update({"kb": kb, "meta": meta})
    return render(request, "uibase/UI.html", context)


