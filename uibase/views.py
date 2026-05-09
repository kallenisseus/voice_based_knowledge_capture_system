"""Django views for the upload + dashboard experience.

This file coordinates HTTP concerns (forms, messages, redirects) while the
AI pipeline and persistence logic stay in `ai/`.
"""

import logging
import os
import json

from django.contrib import messages
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import redirect, render

from .utils import load_knowledge_base


logger = logging.getLogger(__name__)


PLACEHOLDER_CATEGORY_TARGET = 13
DEFAULT_MACHINE_TYPE = "Car"
DEFAULT_MACHINE_NAME = "Car_Brakes"
MACHINE_TYPE_OVERRIDES = {
    "car_brakes": "Car",
    "car brakes": "Car",
}


PLACEHOLDER_CATEGORY_BLUEPRINT = [
    {
        "category": "Car",
        "subcategories": [
            "Car_Brakes",
        ],
    },
    {
        "category": "Boilers",
        "subcategories": [
            "Boiler 01 Main Steam",
            "Boiler 02 Recovery",
            "Boiler 03 Auxiliary",
            "Boiler 04 District Heat",
            "Boiler 05 Peak Load",
            "Boiler 06 Feedwater",
            "Boiler 07 Economizer",
            "Boiler 08 Flue Gas",
        ],
    },
    {
        "category": "Compressors",
        "subcategories": [
            "Compressor 01 Air Intake",
            "Compressor 02 Instrument Air",
            "Compressor 03 Dryer Feed",
            "Compressor 04 Backup Air",
            "Compressor 05 Process Gas",
            "Compressor 06 Booster",
            "Compressor 07 Cooling Loop",
            "Compressor 08 Utility",
        ],
    },
    {
        "category": "Control Systems",
        "subcategories": [
            "DCS Node 01 Boiler Hall",
            "DCS Node 02 Turbine Hall",
            "DCS Node 03 Water Plant",
            "PLC Rack 01 Fuel Line",
            "PLC Rack 02 Conveyor Line",
            "SCADA Gateway 01",
            "SCADA Gateway 02",
            "Historian Server 01",
        ],
    },
    {
        "category": "Conveyors",
        "subcategories": [
            "Conveyor 01 Fuel Infeed",
            "Conveyor 02 Ash Return",
            "Conveyor 03 Slag Discharge",
            "Conveyor 04 Chemical Feed",
            "Conveyor 05 Biomass Feed",
            "Conveyor 06 Lime Feed",
            "Conveyor 07 Storage Transfer",
            "Conveyor 08 Waste Return",
        ],
    },
    {
        "category": "Electrical Distribution",
        "subcategories": [
            "Substation 01 North",
            "Substation 02 South",
            "Switchgear 01 Main Bus",
            "Switchgear 02 Backup Bus",
            "Transformer 01 20kV",
            "Transformer 02 10kV",
            "UPS 01 Control Room",
            "UPS 02 Safety Systems",
        ],
    },
    {
        "category": "Fuel Handling",
        "subcategories": [
            "Fuel Hopper 01",
            "Fuel Hopper 02",
            "Fuel Feeder 01",
            "Fuel Feeder 02",
            "Silo 01 Biomass",
            "Silo 02 Pellets",
            "Ash Handling 01",
            "Ash Handling 02",
        ],
    },
    {
        "category": "Heat Exchangers",
        "subcategories": [
            "HX 01 District Return",
            "HX 02 District Supply",
            "HX 03 Condensate",
            "HX 04 Process Cooling",
            "HX 05 Oil Cooling",
            "HX 06 Stack Recovery",
            "HX 07 Turbine Bypass",
            "HX 08 Utility Loop",
        ],
    },
    {
        "category": "HVAC",
        "subcategories": [
            "AHU 01 Control Room",
            "AHU 02 Turbine Hall",
            "AHU 03 Boiler Hall",
            "Chiller 01",
            "Chiller 02",
            "Vent Fan 01",
            "Vent Fan 02",
            "Exhaust Unit 01",
        ],
    },
    {
        "category": "Instrumentation",
        "subcategories": [
            "Instrument Rack 01",
            "Instrument Rack 02",
            "Flow Meter Train 01",
            "Flow Meter Train 02",
            "Pressure Grid 01",
            "Temperature Grid 01",
            "Gas Analyzer 01",
            "Gas Analyzer 02",
        ],
    },
    {
        "category": "Pumps",
        "subcategories": [
            "Pump 01 Raw Water",
            "Pump 02 Feedwater",
            "Pump 03 Condensate",
            "Pump 04 Cooling Water",
            "Pump 05 District Supply",
            "Pump 06 District Return",
            "Pump 07 Chemical Dosing",
            "Pump 08 Drainage",
        ],
    },
    {
        "category": "Turbines",
        "subcategories": [
            "Turbine 01 Main Unit",
            "Turbine 02 Backup Unit",
            "Turbine 03 Process Unit",
            "Generator 01 Main",
            "Generator 02 Backup",
            "Condenser 01",
            "Condenser 02",
            "Governor 01",
        ],
    },
    {
        "category": "Water Treatment",
        "subcategories": [
            "Filter Line 01",
            "Filter Line 02",
            "RO Train 01",
            "RO Train 02",
            "Degasser 01",
            "Softener 01",
            "Dosing Unit 01",
            "Dosing Unit 02",
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


def _parse_hierarchy_path(raw_value: str) -> list[str]:
    """Parse a free-text hierarchy path into ordered parts."""
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.split(">")]
    return [part for part in parts if part]


def _parse_subcategory_paths(raw_value: str | None) -> list[list[str]]:
    """Parse JSON subcategory paths into normalized list-of-lists."""
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return []

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        parsed = [raw_text]

    if not isinstance(parsed, list):
        return []

    paths = []
    seen = set()
    for item in parsed:
        if isinstance(item, str):
            parts = [part.strip() for part in item.split(">")]
        elif isinstance(item, (list, tuple)):
            parts = [str(value or "").strip() for value in item]
        else:
            continue

        path = [part for part in parts if part]
        if not path:
            continue
        key = tuple(part.casefold() for part in path)
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)

    return paths


def _parse_extra_tags(raw_value: str) -> list[str]:
    """Parse comma-separated tags, preserving order and removing duplicates."""
    raw = str(raw_value or "").strip()
    if not raw:
        return []
    parts = [part.strip() for part in raw.replace(";", ",").split(",")]
    tags = []
    seen = set()
    for part in parts:
        if not part:
            continue
        key = part.casefold()
        if key in seen:
            continue
        seen.add(key)
        tags.append(part)
    return tags


def _parse_keep_summaries_value(raw_value: str | None) -> bool:
    """Parse checkbox-like form values into a boolean."""
    value = (raw_value or "").strip().lower()
    return value not in {"0", "false", "off", "no"}


def _is_delete_confirmed(request) -> bool:
    """Require explicit confirmation token for destructive actions."""
    return (request.POST.get("confirm_delete") or "").strip().lower() == "yes"


def _parse_optional_int(raw_value: str | None) -> int | None:
    """Parse optional form integer values safely."""
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _normalize_hex_color(value: str | None, fallback: str = "#3A78F2") -> str:
    """Normalize color inputs to #RRGGBB format."""
    text = str(value or "").strip()
    if not text:
        return fallback
    if text.startswith("#"):
        text = text[1:]
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        return fallback
    try:
        int(text, 16)
    except ValueError:
        return fallback
    return f"#{text.upper()}"


def _resolve_machine_type(machine_type: str | None, machine_name: str | None) -> str:
    """Normalize type and auto-map known machine names into stable type buckets."""
    machine_name_value = (machine_name or "").strip()
    machine_type_value = (machine_type or "").strip()
    if machine_type_value and machine_type_value.casefold() != "unassigned":
        return machine_type_value

    override = MACHINE_TYPE_OVERRIDES.get(machine_name_value.casefold())
    if override:
        return override

    return machine_type_value or DEFAULT_MACHINE_TYPE


def _build_taxonomy_options(uploads: list[dict]) -> tuple[list[str], list[str], dict]:
    """Collect known type/machine suggestions from placeholders + stored uploads."""
    type_options = {blueprint["category"] for blueprint in PLACEHOLDER_CATEGORY_BLUEPRINT}
    machine_options = {
        machine_name
        for blueprint in PLACEHOLDER_CATEGORY_BLUEPRINT
        for machine_name in blueprint.get("subcategories", [])
    }
    machine_catalog = {
        blueprint["category"]: sorted(blueprint.get("subcategories", []), key=lambda value: str(value).casefold())
        for blueprint in PLACEHOLDER_CATEGORY_BLUEPRINT
    }

    for upload in uploads or []:
        machine_name = (upload.get("machine_name") or "").strip() or (upload.get("category") or "").strip()
        machine_type = _resolve_machine_type(upload.get("machine_type"), machine_name)
        type_options.add(machine_type)
        if machine_name:
            machine_options.add(machine_name)
            machine_catalog.setdefault(machine_type, [])
            if machine_name not in machine_catalog[machine_type]:
                machine_catalog[machine_type].append(machine_name)

    for machine_type in machine_catalog:
        machine_catalog[machine_type] = sorted(
            machine_catalog[machine_type],
            key=lambda value: str(value).casefold(),
        )

    return (
        sorted(type_options, key=lambda value: str(value).casefold()),
        sorted(machine_options, key=lambda value: str(value).casefold()),
        machine_catalog,
    )


def _process_uploaded_audio(
    uploaded,
    *,
    author_name: str,
    machine_name: str,
    machine_type: str,
    subcategory_paths: list[list[str]],
    hierarchy_path: list[str],
    extra_tags: list[str],
    subcategory_paths_raw: str,
    hierarchy_path_raw: str,
    extra_tags_raw: str,
) -> dict:
    """Persist an uploaded file and trigger pipeline processing."""
    machine_name = (machine_name or "").strip()
    machine_type = _resolve_machine_type(machine_type, machine_name)

    submitted_payload = {
        "machine_name_submitted": machine_name,
        "machine_type_submitted": machine_type,
        "subcategory_paths_submitted": subcategory_paths_raw,
        "hierarchy_path_submitted": hierarchy_path_raw,
        "extra_tags_submitted": extra_tags_raw,
        "author_submitted": author_name,
        # Keep legacy template key usage stable in case any UI code still references it.
        "category_submitted": machine_name,
    }

    if not machine_name:
        return {
            "post_success": False,
            "error": "Machine name is required.",
            **submitted_payload,
        }

    if not uploaded:
        return {
            "post_success": False,
            "error": "No file uploaded.",
            **submitted_payload,
        }

    target_dir = _ensure_audio_dir()
    if not target_dir:
        return {
            "post_success": False,
            "error": "MEDIA_ROOT is not configured.",
            **submitted_payload,
        }

    fs = FileSystemStorage(location=target_dir)
    saved_name = fs.save(uploaded.name, uploaded)

    try:
        from ai.pipeline import process_uploaded_file

        process_uploaded_file(
            audio_file=saved_name,
            author=author_name,
            machine_name=machine_name,
            machine_type=machine_type,
            subcategory_paths=subcategory_paths,
            hierarchy_path=hierarchy_path,
            extra_tags=extra_tags,
        )
    except Exception as exc:
        logger.exception("Failed to process uploaded audio '%s'", saved_name)
        return {
            "post_success": False,
            "error": f"Upload saved, but processing failed: {exc}",
            "saved_file": saved_name,
            **submitted_payload,
        }

    return {
        "post_success": True,
        "saved_file": saved_name,
        **submitted_payload,
    }


def _build_dashboard_context() -> dict:
    """Load the current dashboard data snapshot for template rendering."""
    raw = load_knowledge_base() or {}
    kb = raw.get("data", {}) if isinstance(raw, dict) else {}
    meta = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    uploads = raw.get("uploads", []) if isinstance(raw, dict) else []
    styles = raw.get("styles", {}) if isinstance(raw, dict) else {}
    machine_type_options, machine_name_options, machine_catalog = _build_taxonomy_options(uploads)
    default_machine_type = (
        DEFAULT_MACHINE_TYPE
        if DEFAULT_MACHINE_TYPE in machine_type_options
        else (machine_type_options[0] if machine_type_options else DEFAULT_MACHINE_TYPE)
    )
    scoped_machine_names = machine_catalog.get(default_machine_type, [])
    if DEFAULT_MACHINE_NAME in scoped_machine_names:
        default_machine_name = DEFAULT_MACHINE_NAME
    elif scoped_machine_names:
        default_machine_name = scoped_machine_names[0]
    elif machine_name_options:
        default_machine_name = machine_name_options[0]
    else:
        default_machine_name = DEFAULT_MACHINE_NAME

    kb = _inject_placeholder_categories(kb)
    kb = _sort_knowledge_base(kb)
    return {
        "kb": kb,
        "meta": meta,
        "uploads": uploads,
        "styles": styles,
        "machine_type_options": machine_type_options,
        "machine_name_options": machine_name_options,
        "machine_catalog": machine_catalog,
        # Keep submitted-form fields defined even on GET requests.
        "machine_name_submitted": default_machine_name,
        "machine_type_submitted": default_machine_type,
        "subcategory_paths_submitted": "[]",
        "hierarchy_path_submitted": "",
        "extra_tags_submitted": "",
        "author_submitted": "",
        "category_submitted": default_machine_name,
    }


def dashboard(request):
    """Render dashboard and handle upload submissions from the same page."""
    context = _build_dashboard_context()

    if request.method == "POST":
        uploaded = request.FILES.get("audio_file")
        machine_name = request.POST.get("machine_name", "").strip()
        machine_type = _resolve_machine_type(
            request.POST.get("machine_type", "").strip(),
            machine_name,
        )
        # New upload flow is tags-first: subcategory/hierarchy are no longer
        # entered manually and are intentionally forced empty for new files.
        subcategory_paths_raw = "[]"
        subcategory_paths = []
        hierarchy_path_raw = ""
        hierarchy_path = []
        extra_tags_raw = request.POST.get("extra_tags", "").strip()
        extra_tags = _parse_extra_tags(extra_tags_raw)
        author_name = request.POST.get("author_name", "").strip()

        submission_result = _process_uploaded_audio(
            uploaded,
            author_name=author_name,
            machine_name=machine_name,
            machine_type=machine_type,
            subcategory_paths=subcategory_paths,
            hierarchy_path=hierarchy_path,
            extra_tags=extra_tags,
            subcategory_paths_raw=subcategory_paths_raw,
            hierarchy_path_raw=hierarchy_path_raw,
            extra_tags_raw=extra_tags_raw,
        )
        context.update(submission_result)
        context.update(_build_dashboard_context())
        context.update(
            {
                key: value
                for key, value in submission_result.items()
                if key in {"post_success", "error", "saved_file"} or key.endswith("_submitted")
            }
        )

    return render(request, "uibase/UI.html", context)


def delete_upload(request, upload_id: int):
    """Delete one uploaded file and all related transcript/cluster data."""
    if request.method != "POST":
        return redirect("dashboard")
    if not _is_delete_confirmed(request):
        messages.warning(request, "Deletion cancelled: confirmation was not provided.")
        return redirect("dashboard")

    try:
        from ai.pipeline import delete_uploaded_audio

        keep_summaries = _parse_keep_summaries_value(request.POST.get("keep_summaries", "1"))
        file_name, warning = delete_uploaded_audio(upload_id, keep_summaries=keep_summaries)
        if warning:
            messages.warning(request, warning)
        elif keep_summaries:
            messages.success(
                request,
                f"Deleted {file_name} and all related data. Affected summaries were marked stale for manual redo.",
            )
        else:
            messages.success(
                request,
                f"Deleted {file_name} and all related data. Related summaries were recomputed.",
            )
    except Exception as exc:
        logger.exception("Failed to delete upload %s", upload_id)
        messages.error(request, f"Failed to delete file: {exc}")

    return redirect("dashboard")


def bulk_delete_uploads(request):
    """Delete multiple uploads in one action with optional stale-summary mode."""
    if request.method != "POST":
        return redirect("dashboard")
    if not _is_delete_confirmed(request):
        messages.warning(request, "Deletion cancelled: confirmation was not provided.")
        return redirect("dashboard")

    upload_ids = request.POST.getlist("upload_ids")
    keep_summaries = _parse_keep_summaries_value(request.POST.get("keep_summaries", "1"))

    try:
        from ai.pipeline import delete_uploaded_audio_batch

        payload = delete_uploaded_audio_batch(upload_ids, keep_summaries=keep_summaries)
        deleted_count = int(payload.get("deleted_count") or 0)
        deleted_files = payload.get("deleted_files") or []
        affected_categories = payload.get("affected_categories") or []
        dropped_categories = payload.get("dropped_categories") or []
        stale_clusters = int(payload.get("stale_clusters") or 0)
        refreshed_categories = payload.get("refreshed_categories") or []
        warning = payload.get("warning")

        if deleted_count <= 0:
            messages.warning(request, "No files were selected for deletion.")
            return redirect("dashboard")

        file_preview = ", ".join(deleted_files[:4])
        if len(deleted_files) > 4:
            file_preview = f"{file_preview}, +{len(deleted_files) - 4} more"
        category_preview = ", ".join(affected_categories) if affected_categories else "none"

        if warning:
            messages.warning(request, warning)
        elif keep_summaries:
            messages.success(
                request,
                (
                    f"Deleted {deleted_count} files ({file_preview}) and all related data. "
                    f"Affected categories: {category_preview}. "
                    f"Marked {stale_clusters} undercategories as stale. "
                    f"Fully removed categories: {', '.join(dropped_categories) if dropped_categories else 'none'}."
                ),
            )
        else:
            messages.success(
                request,
                (
                    f"Deleted {deleted_count} files ({file_preview}) and all related data. "
                    f"Affected categories: {category_preview}. "
                    f"Recomputed summaries in categories: "
                    f"{', '.join(refreshed_categories) if refreshed_categories else 'none'}."
                ),
            )
    except Exception as exc:
        logger.exception("Failed bulk deletion for uploads: %s", upload_ids)
        messages.error(request, f"Failed to delete selected files: {exc}")

    return redirect("dashboard")


def redo_category_summary(request):
    """Regenerate all summaries under one machine/category bucket."""
    if request.method != "POST":
        return redirect("dashboard")

    category = (request.POST.get("category") or "").strip()
    if not category:
        messages.error(request, "Missing category for re-summary.")
        return redirect("dashboard")

    try:
        from ai.pipeline import redo_category_summaries

        payload = redo_category_summaries(category)
        updated = int(payload.get("updated") or 0)
        deleted = int(payload.get("deleted") or 0)
        messages.success(
            request,
            f"Redid summaries for {category}: updated {updated}, removed empty {deleted}.",
        )
    except Exception as exc:
        logger.exception("Failed to redo category summaries for %s", category)
        messages.error(request, f"Failed to redo category summaries: {exc}")

    return redirect("dashboard")


def redo_cluster_summary_view(request, cluster_id: int):
    """Regenerate one undercategory summary from current remaining snippets."""
    if request.method != "POST":
        return redirect("dashboard")

    try:
        from ai.pipeline import redo_cluster_summary

        payload = redo_cluster_summary(cluster_id)
        updated = int(payload.get("updated") or 0)
        deleted = int(payload.get("deleted") or 0)
        category = payload.get("category") or "Category"
        cluster_name = payload.get("cluster_name") or "Undercategory"

        if deleted:
            messages.success(
                request,
                f"{cluster_name} in {category} had no remaining data and was removed.",
            )
        elif updated:
            messages.success(
                request,
                f"Redid summary for {cluster_name} in {category}.",
            )
        else:
            messages.warning(request, "No matching undercategory found to re-summarize.")
    except Exception as exc:
        logger.exception("Failed to redo cluster summary for %s", cluster_id)
        messages.error(request, f"Failed to redo undercategory summary: {exc}")

    return redirect("dashboard")


def remove_category(request):
    """Remove one overcategory and all files/data that belong to it."""
    if request.method != "POST":
        return redirect("dashboard")
    if not _is_delete_confirmed(request):
        messages.warning(request, "Category removal cancelled: confirmation was not provided.")
        return redirect("dashboard")

    machine_type = (request.POST.get("machine_type") or "").strip()
    if not machine_type:
        messages.error(request, "Missing category to remove.")
        return redirect("dashboard")

    try:
        from ai.pipeline import remove_machine_type_category

        payload = remove_machine_type_category(machine_type)
        deleted_count = int(payload.get("deleted_count") or 0)
        deleted_files = payload.get("deleted_files") or []
        warning = payload.get("warning")

        if warning:
            messages.warning(request, warning)
        elif deleted_count <= 0:
            messages.warning(request, f"No files found under category {machine_type}.")
        else:
            preview = ", ".join(deleted_files[:4])
            if len(deleted_files) > 4:
                preview = f"{preview}, +{len(deleted_files) - 4} more"
            messages.success(
                request,
                f"Removed category {machine_type}. Deleted {deleted_count} files and all related data ({preview}).",
            )
    except Exception as exc:
        logger.exception("Failed to remove category %s", machine_type)
        messages.error(request, f"Failed to remove category: {exc}")

    return redirect("dashboard")


def remove_cluster_view(request, cluster_id: int):
    """Remove one undercategory cluster from the knowledge view."""
    if request.method != "POST":
        return redirect("dashboard")
    if not _is_delete_confirmed(request):
        messages.warning(request, "Undercategory removal cancelled: confirmation was not provided.")
        return redirect("dashboard")

    try:
        from ai.pipeline import remove_cluster

        payload = remove_cluster(cluster_id)
        deleted = int(payload.get("deleted") or 0)
        cluster_name = payload.get("cluster_name") or "Undercategory"
        category = payload.get("category") or "Category"
        member_count = int(payload.get("member_count") or 0)
        files = payload.get("files") or []

        if deleted <= 0:
            messages.warning(request, "No matching undercategory found to remove.")
        else:
            file_preview = ", ".join(files[:4]) if files else "no linked files"
            if len(files) > 4:
                file_preview = f"{file_preview}, +{len(files) - 4} more"
            messages.success(
                request,
                (
                    f"Removed undercategory {cluster_name} in {category}. "
                    f"Detached {member_count} snippets. Linked files: {file_preview}."
                ),
            )
    except Exception as exc:
        logger.exception("Failed to remove cluster %s", cluster_id)
        messages.error(request, f"Failed to remove undercategory: {exc}")

    return redirect("dashboard")


def rename_category_view(request):
    """Rename one top-level type/category across uploads and styles."""
    if request.method != "POST":
        return redirect("dashboard")

    machine_type = (request.POST.get("machine_type") or "").strip()
    new_machine_type = (request.POST.get("new_machine_type") or "").strip()
    if not machine_type or not new_machine_type:
        messages.error(request, "Both current and new category names are required.")
        return redirect("dashboard")

    try:
        from ai.pipeline import rename_machine_type
        from uibase.models import MachineTypeStyle

        payload = rename_machine_type(machine_type, new_machine_type)
        warning = payload.get("warning")
        updated = int(payload.get("updated") or 0)
        if warning:
            messages.warning(request, warning)
            return redirect("dashboard")
        if updated <= 0:
            messages.warning(request, f"No items were renamed from {machine_type}.")
            return redirect("dashboard")

        old_style = MachineTypeStyle.objects.filter(machine_type=machine_type).first()
        if old_style:
            existing_target = MachineTypeStyle.objects.filter(machine_type=new_machine_type).first()
            if existing_target and existing_target.pk != old_style.pk:
                old_color = _normalize_hex_color(old_style.color_hex)
                target_color = _normalize_hex_color(existing_target.color_hex)
                if target_color == "#3A78F2" and old_color != "#3A78F2":
                    existing_target.color_hex = old_color
                    existing_target.save(update_fields=["color_hex"])
                old_style.delete()
            else:
                old_style.machine_type = new_machine_type
                old_style.save(update_fields=["machine_type"])

        messages.success(
            request,
            f"Renamed category {machine_type} to {new_machine_type} ({updated} uploads updated).",
        )
    except Exception as exc:
        logger.exception("Failed to rename category %s to %s", machine_type, new_machine_type)
        messages.error(request, f"Failed to rename category: {exc}")

    return redirect("dashboard")


def set_machine_type_color_view(request):
    """Persist UI color for one top-level type/category."""
    if request.method != "POST":
        return redirect("dashboard")

    machine_type = (request.POST.get("machine_type") or "").strip()
    if not machine_type:
        messages.error(request, "Missing category for color update.")
        return redirect("dashboard")

    color_hex = _normalize_hex_color(request.POST.get("color_hex"), fallback="#3A78F2")
    try:
        from uibase.models import MachineTypeStyle

        MachineTypeStyle.objects.update_or_create(
            machine_type=machine_type,
            defaults={"color_hex": color_hex},
        )
        messages.success(request, f"Updated category color for {machine_type}.")
    except Exception as exc:
        logger.exception("Failed to set category color for %s", machine_type)
        messages.error(request, f"Failed to update category color: {exc}")

    return redirect("dashboard")


def set_machine_color_view(request):
    """Persist UI color for one machine."""
    if request.method != "POST":
        return redirect("dashboard")

    machine_name = (request.POST.get("machine_name") or "").strip()
    if not machine_name:
        messages.error(request, "Missing machine for color update.")
        return redirect("dashboard")

    color_hex = _normalize_hex_color(request.POST.get("color_hex"), fallback="#3A78F2")
    try:
        from uibase.models import MachineStyle

        MachineStyle.objects.update_or_create(
            machine_name=machine_name,
            defaults={"color_hex": color_hex},
        )
        messages.success(request, f"Updated machine color for {machine_name}.")
    except Exception as exc:
        logger.exception("Failed to set machine color for %s", machine_name)
        messages.error(request, f"Failed to update machine color: {exc}")

    return redirect("dashboard")


def rename_cluster_view(request, cluster_id: int):
    """Rename one undercategory cluster."""
    if request.method != "POST":
        return redirect("dashboard")

    new_name = (request.POST.get("new_cluster_name") or "").strip()
    if not new_name:
        messages.error(request, "New undercategory name is required.")
        return redirect("dashboard")

    try:
        from ai.pipeline import rename_cluster

        payload = rename_cluster(cluster_id, new_name)
        warning = payload.get("warning")
        updated = int(payload.get("updated") or 0)
        old_name = payload.get("old_name") or "Undercategory"
        final_name = payload.get("new_name") or new_name
        category = payload.get("category") or "Category"

        if warning:
            messages.warning(request, warning)
        elif updated <= 0:
            messages.warning(request, "No undercategory was renamed.")
        else:
            messages.success(
                request,
                f"Renamed undercategory {old_name} to {final_name} in {category}.",
            )
    except Exception as exc:
        logger.exception("Failed to rename cluster %s", cluster_id)
        messages.error(request, f"Failed to rename undercategory: {exc}")

    return redirect("dashboard")


def merge_cluster_view(request, cluster_id: int):
    """Move all snippets from one undercategory into a selected destination."""
    if request.method != "POST":
        return redirect("dashboard")

    target_cluster_id = _parse_optional_int(request.POST.get("target_cluster_id"))
    new_cluster_name = (request.POST.get("new_cluster_name") or "").strip()
    force_uncategorized = (request.POST.get("force_uncategorized") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        from ai.pipeline import merge_cluster_into_target

        payload = merge_cluster_into_target(
            cluster_id=cluster_id,
            target_cluster_id=target_cluster_id,
            new_cluster_name=new_cluster_name,
            force_uncategorized=force_uncategorized,
        )
        warning = payload.get("warning")
        updated = int(payload.get("updated") or 0)
        moved_segments = int(payload.get("moved_segments") or 0)
        source_name = payload.get("source_name") or "Undercategory"
        target_name = payload.get("target_name") or "Undercategory"
        category = payload.get("category") or "Category"

        if warning:
            messages.warning(request, warning)
        elif updated <= 0 and moved_segments <= 0:
            messages.warning(request, "No snippets were moved.")
        else:
            messages.success(
                request,
                (
                    f"Moved {moved_segments} snippets from {source_name} to {target_name} "
                    f"in {category}."
                ),
            )
    except Exception as exc:
        logger.exception("Failed to merge cluster %s", cluster_id)
        messages.error(request, f"Failed to move undercategory snippets: {exc}")

    return redirect("dashboard")


def move_segment_view(request, segment_id: int):
    """Move one snippet to another undercategory."""
    if request.method != "POST":
        return redirect("dashboard")

    target_cluster_id = _parse_optional_int(request.POST.get("target_cluster_id"))
    new_cluster_name = (request.POST.get("new_cluster_name") or "").strip()
    force_uncategorized = (request.POST.get("force_uncategorized") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    try:
        from ai.pipeline import move_segment_to_cluster

        payload = move_segment_to_cluster(
            segment_id=segment_id,
            target_cluster_id=target_cluster_id,
            new_cluster_name=new_cluster_name,
            force_uncategorized=force_uncategorized,
        )
        warning = payload.get("warning")
        updated = int(payload.get("updated") or 0)
        target_name = payload.get("target_cluster_name") or "Undercategory"
        category = payload.get("category") or "Category"
        updated_clusters = int(payload.get("updated_clusters") or 0)
        deleted_clusters = int(payload.get("deleted_clusters") or 0)

        if warning:
            messages.warning(request, warning)
        elif updated <= 0:
            messages.warning(request, "No snippet was moved.")
        else:
            messages.success(
                request,
                (
                    f"Moved snippet to {target_name} in {category}. "
                    f"Refreshed {updated_clusters} summaries and removed {deleted_clusters} empty undercategories."
                ),
            )
    except Exception as exc:
        logger.exception("Failed to move segment %s", segment_id)
        messages.error(request, f"Failed to move snippet: {exc}")

    return redirect("dashboard")


def delete_segment_view(request, segment_id: int):
    """Delete one snippet from knowledge base and refresh affected summaries."""
    if request.method != "POST":
        return redirect("dashboard")
    if not _is_delete_confirmed(request):
        messages.warning(request, "Snippet deletion cancelled: confirmation was not provided.")
        return redirect("dashboard")

    try:
        from ai.pipeline import delete_segment_from_knowledge

        payload = delete_segment_from_knowledge(segment_id)
        warning = payload.get("warning")
        deleted = int(payload.get("deleted") or 0)
        category = payload.get("category") or "Category"
        updated_clusters = int(payload.get("updated_clusters") or 0)
        deleted_clusters = int(payload.get("deleted_clusters") or 0)

        if warning:
            messages.warning(request, warning)
        elif deleted <= 0:
            messages.warning(request, "No snippet was deleted.")
        else:
            messages.success(
                request,
                (
                    f"Deleted snippet from {category}. "
                    f"Refreshed {updated_clusters} summaries and removed {deleted_clusters} empty undercategories."
                ),
            )
    except Exception as exc:
        logger.exception("Failed to delete segment %s", segment_id)
        messages.error(request, f"Failed to delete snippet: {exc}")

    return redirect("dashboard")


