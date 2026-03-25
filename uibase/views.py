"""Django views for the upload + dashboard experience.

This file coordinates HTTP concerns (forms, messages, redirects) while the
AI pipeline and persistence logic stay in `ai/`.
"""

import logging
import os

from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect, render

from .utils import load_knowledge_base


logger = logging.getLogger(__name__)


PLACEHOLDER_CATEGORY_TARGET = 12


PLACEHOLDER_CATEGORY_BLUEPRINT = [
    {
        "category": "Safety & Compliance",
        "subcategories": [
            "PPE Checks",
            "Lockout / Tagout",
            "Risk Assessment",
            "Permit Verification",
        ],
    },
    {
        "category": "Hydraulic Systems",
        "subcategories": [
            "Pump Diagnostics",
            "Hose & Leak Inspection",
            "Pressure Testing",
            "Valve Calibration",
        ],
    },
    {
        "category": "Electrical Diagnostics",
        "subcategories": [
            "Battery & Charging",
            "Sensor Validation",
            "Harness Continuity",
            "Ground Fault Tracing",
            "Control Module Checks",
        ],
    },
    {
        "category": "Mechanical Service",
        "subcategories": [
            "Bearing Inspection",
            "Alignment & Balancing",
            "Fastener Torque Control",
            "Wear Pattern Review",
        ],
    },
    {
        "category": "Process Quality",
        "subcategories": [
            "Root Cause Review",
            "Verification Run",
            "Documentation Standards",
        ],
    },
    {
        "category": "Operations Handover",
        "subcategories": [
            "Job Briefing",
            "Spare Parts Status",
            "Follow-up Actions",
            "Customer Communication",
        ],
    },
    {
        "category": "Cooling & Lubrication",
        "subcategories": [
            "Coolant Flow Checks",
            "Oil Condition Review",
            "Filter Service",
            "Temperature Stability",
        ],
    },
    {
        "category": "Pneumatic Controls",
        "subcategories": [
            "Air Supply Integrity",
            "Pressure Regulation",
            "Actuator Response",
        ],
    },
    {
        "category": "Drive Train Health",
        "subcategories": [
            "Belt & Chain Tension",
            "Gearbox Noise Analysis",
            "Shaft Alignment",
            "Load Transfer Checks",
        ],
    },
    {
        "category": "Calibration & Setup",
        "subcategories": [
            "Reference Baseline",
            "Parameter Tuning",
            "Tolerance Verification",
            "Post-Setup Validation",
        ],
    },
    {
        "category": "Incident Follow-up",
        "subcategories": [
            "Fault Timeline",
            "Contributing Factors",
            "Corrective Actions",
            "Prevention Measures",
            "Owner Assignment",
        ],
    },
    {
        "category": "Inventory & Parts",
        "subcategories": [
            "Critical Spare Status",
            "Lead Time Risks",
            "Replacement Mapping",
            "Usage Patterns",
        ],
    },
]


def _placeholder_subcategory_payload(category_name: str, subcategory_name: str) -> dict:
    """Small starter payload shown while the knowledge base is still sparse."""
    return {
        "summary": (
            f"Starter template for {subcategory_name}. This section will be replaced "
            "automatically as real snippets are added."
        ),
        "summary_sections": [
            {
                "title": "What To Capture",
                "body": (
                    f"Collect key observations and decisions under {subcategory_name} "
                    f"to build a useful summary for {category_name}."
                ),
            }
        ],
        "windows": [],
    }


def _inject_placeholder_categories(kb: dict) -> dict:
    """Add placeholder top categories while real categories are still limited."""
    real_kb = dict(kb or {})
    if len(real_kb) >= PLACEHOLDER_CATEGORY_TARGET:
        return real_kb

    augmented = dict(real_kb)
    for blueprint in PLACEHOLDER_CATEGORY_BLUEPRINT:
        category_name = blueprint["category"]
        if category_name in augmented:
            continue

        augmented[category_name] = {
            subcategory: _placeholder_subcategory_payload(category_name, subcategory)
            for subcategory in blueprint["subcategories"]
        }

        if len(augmented) >= PLACEHOLDER_CATEGORY_TARGET:
            break

    return augmented


def _sort_knowledge_base(kb: dict) -> dict:
    """Return category/subcategory dictionaries sorted alphabetically."""
    sorted_kb = {}
    for category_name in sorted((kb or {}).keys(), key=lambda value: str(value).casefold()):
        subcategories = kb.get(category_name) or {}
        if isinstance(subcategories, dict):
            sorted_subcategories = {
                subcategory_name: subcategories[subcategory_name]
                for subcategory_name in sorted(subcategories.keys(), key=lambda value: str(value).casefold())
            }
        else:
            sorted_subcategories = subcategories
        sorted_kb[category_name] = sorted_subcategories
    return sorted_kb


def _ensure_audio_dir():
    """Create and return the media/audio directory used by uploaded files."""
    media_root = getattr(settings, "MEDIA_ROOT", None)
    if not media_root:
        return None
    audio_dir = os.path.join(media_root, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    return audio_dir


def _process_uploaded_audio(uploaded, author_name: str, category: str) -> dict:
    """Persist an uploaded file and trigger pipeline processing."""
    if not uploaded:
        return {
            "post_success": False,
            "error": "No file uploaded.",
            "category_submitted": category,
            "author_submitted": author_name,
        }

    target_dir = _ensure_audio_dir()
    if not target_dir:
        return {
            "post_success": False,
            "error": "MEDIA_ROOT is not configured.",
            "category_submitted": category,
            "author_submitted": author_name,
        }

    fs = FileSystemStorage(location=target_dir)
    saved_name = fs.save(uploaded.name, uploaded)

    try:
        from ai.pipeline import process_uploaded_file

        process_uploaded_file(saved_name, author_name, category)
    except Exception as exc:
        logger.exception("Failed to process uploaded audio '%s'", saved_name)
        return {
            "post_success": False,
            "error": f"Upload saved, but processing failed: {exc}",
            "saved_file": saved_name,
            "category_submitted": category,
            "author_submitted": author_name,
        }

    return {
        "post_success": True,
        "saved_file": saved_name,
        "category_submitted": category,
        "author_submitted": author_name,
    }


def _build_dashboard_context() -> dict:
    """Load the current dashboard data snapshot for template rendering."""
    raw = load_knowledge_base() or {}
    kb = raw.get("data", {}) if isinstance(raw, dict) else {}
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    uploads = raw.get("uploads", []) if isinstance(raw, dict) else []
    kb = _inject_placeholder_categories(kb)
    kb = _sort_knowledge_base(kb)
    return {"kb": kb, "meta": meta, "uploads": uploads}


def dashboard(request):
    """Render dashboard and handle upload submissions from the same page."""
    context = _build_dashboard_context()

    if request.method == "POST":
        uploaded = request.FILES.get("audio_file")
        category = request.POST.get("category", "").strip()
        author_name = request.POST.get("author_name", "").strip()

        context.update(_process_uploaded_audio(uploaded, author_name, category))
        context.update(_build_dashboard_context())

    return render(request, "uibase/UI.html", context)


def delete_upload(request, upload_id: int):
    """Delete one uploaded file and all related transcript/cluster data."""
    if request.method != "POST":
        return redirect("dashboard")

    try:
        from ai.pipeline import delete_uploaded_audio

        file_name, warning = delete_uploaded_audio(upload_id)
        if warning:
            messages.warning(request, warning)
        else:
            messages.success(request, f"Deleted {file_name} and all related data.")
    except Exception as exc:
        logger.exception("Failed to delete upload %s", upload_id)
        messages.error(request, f"Failed to delete file: {exc}")

    return redirect("dashboard")


