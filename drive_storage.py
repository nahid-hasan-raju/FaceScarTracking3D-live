"""
drive_storage.py
=================
Replaces the local-filesystem dataset walk (`Path.iterdir()` etc.) with an
equivalent walk over a Google Drive folder tree, using a service account.

Expected Drive layout (mirrors the old local layout exactly):

    <ROOT_FOLDER_ID>/
        <patient>/
            <timepoint>/
                <scan_id>/
                    <scan_id>.tif
                    <scan_id>_burn_polygons.json   (optional)

SETUP (see README_DEPLOY.md for the full walkthrough):
    1. Create a Google Cloud service account, enable the Drive API.
    2. Share your Drive dataset folder with the service account's email
       (Editor access) so it can read/write.
    3. Set two environment variables on the host:
         GOOGLE_CREDENTIALS_JSON  -> the full service account JSON key, as one string
         DRIVE_ROOT_FOLDER_ID     -> the Drive folder ID of the dataset root

Caching: walking the whole Drive tree is one API call per folder, which
gets slow on every request for a large dataset. The tree is cached in
memory for TREE_TTL_SECONDS and rebuilt lazily after that, or immediately
after a save (see invalidate()).
"""

import io
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

SCOPES = ["https://www.googleapis.com/auth/drive"]
FOLDER_MIME = "application/vnd.google-apps.folder"
DAY0_NAMES = {"D00", "D0", "DAY0", "Day0"}
TREE_TTL_SECONDS = 300
MAX_WORKERS = 10  # concurrent Drive API calls during a tree walk

_service = None
_CACHE = {"tree": None, "ts": 0}
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Auth / low-level Drive helpers
# ---------------------------------------------------------------------------

def _build_service():
    creds_json = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_FILE")

    if creds_json:
        info = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    elif creds_path:
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    else:
        raise RuntimeError(
            "No Google credentials found. Set GOOGLE_CREDENTIALS_JSON (the service "
            "account key as one JSON string) or GOOGLE_CREDENTIALS_FILE (a path)."
        )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def get_service():
    """One Drive service instance per thread — the googleapiclient http
    object underneath isn't safe to share across threads, and the tree
    walk below runs many folder listings concurrently."""
    svc = getattr(_thread_local, "service", None)
    if svc is None:
        svc = _build_service()
        _thread_local.service = svc
    return svc


def _list_children(folder_id):
    """Direct children of folder_id: list of {id, name, mimeType}."""
    service = get_service()
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def download_file_bytes(file_id: str) -> bytes:
    service = get_service()
    req = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def upload_json(file_id, parent_id, filename, data: dict):
    """Create a new json file (file_id=None) or overwrite an existing one."""
    service = get_service()
    content = json.dumps(data, indent=2).encode("utf-8")
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="application/json", resumable=False)
    if file_id:
        return service.files().update(fileId=file_id, media_body=media, supportsAllDrives=True).execute()
    meta = {"name": filename, "parents": [parent_id]}
    return service.files().create(
        body=meta, media_body=media, fields="id, name", supportsAllDrives=True
    ).execute()


def copy_file(file_id, new_name, parent_id):
    service = get_service()
    body = {"name": new_name, "parents": [parent_id]}
    return service.files().copy(fileId=file_id, body=body, supportsAllDrives=True).execute()


# ---------------------------------------------------------------------------
# Tree walk + cache
# ---------------------------------------------------------------------------

def _is_day0(name: str) -> bool:
    return name.upper().replace("-", "") in {n.upper() for n in DAY0_NAMES}


def _scan_from_folder(scan_folder: dict):
    """Fetch one scan folder's files and return (scan_id, scan_dict) or None."""
    files = _list_children(scan_folder["id"])
    tif = next(
        (f for f in files
         if f["name"].lower().endswith(".tif") and "_seg" not in f["name"].lower()),
        None,
    )
    if not tif:
        return None
    scan_id = tif["name"].rsplit(".", 1)[0]
    json_file = next((f for f in files if f["name"] == f"{scan_id}_burn_polygons.json"), None)
    return scan_id, {"id": scan_folder["id"], "tif": tif, "json": json_file}


def _timepoint_from_folder(tp_folder: dict):
    """Fetch one timepoint folder's scans (in parallel) and return (name, dict)."""
    scan_folders = [c for c in _list_children(tp_folder["id"]) if c["mimeType"] == FOLDER_MIME]
    scan_map = {}
    if scan_folders:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for result in pool.map(_scan_from_folder, scan_folders):
                if result:
                    scan_id, scan_dict = result
                    scan_map[scan_id] = scan_dict
    return tp_folder["name"], {
        "id": tp_folder["id"],
        "is_day0": _is_day0(tp_folder["name"]),
        "scans": scan_map,
    }


def _patient_from_folder(patient_folder: dict):
    """Fetch one patient folder's timepoints (in parallel) and return (name, dict)."""
    tp_folders = [c for c in _list_children(patient_folder["id"]) if c["mimeType"] == FOLDER_MIME]
    tp_map = {}
    if tp_folders:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for tp_name, tp_dict in pool.map(_timepoint_from_folder, tp_folders):
                tp_map[tp_name] = tp_dict
    return patient_folder["name"], {"id": patient_folder["id"], "timepoints": tp_map}


def _build_tree(root_id: str) -> dict:
    patient_folders = [c for c in _list_children(root_id) if c["mimeType"] == FOLDER_MIME]
    tree = {}
    if patient_folders:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for patient_name, patient_dict in pool.map(_patient_from_folder, patient_folders):
                tree[patient_name] = patient_dict
    return tree


def get_tree(root_id: str, force: bool = False) -> dict:
    now = time.time()
    if force or _CACHE["tree"] is None or (now - _CACHE["ts"]) > TREE_TTL_SECONDS:
        _CACHE["tree"] = _build_tree(root_id)
        _CACHE["ts"] = now
    return _CACHE["tree"]


def invalidate():
    _CACHE["ts"] = 0


def find_scan(tree: dict, scan_id: str):
    """Return (patient_name, timepoint_name, scan_dict) or None."""
    for patient_name, patient in tree.items():
        for tp_name, tp in patient["timepoints"].items():
            if scan_id in tp["scans"]:
                return patient_name, tp_name, tp["scans"][scan_id]
    return None


# ---------------------------------------------------------------------------
# High-level listing functions (mirror the old local-filesystem versions)
# ---------------------------------------------------------------------------

def discover_patients(tree: dict):
    patients = []
    for patient_name, patient in sorted(tree.items()):
        scan_count = 0
        day0_count = 0
        polygons_saved = 0
        for tp in patient["timepoints"].values():
            for scan in tp["scans"].values():
                scan_count += 1
                if tp["is_day0"]:
                    day0_count += 1
                if scan["json"] is not None:
                    polygons_saved += 1
        patients.append({
            "patient": patient_name,
            "timepoint_count": len(patient["timepoints"]),
            "scan_count": scan_count,
            "day0_count": day0_count,
            "polygons_saved": polygons_saved,
        })
    return patients


def discover_timepoints(tree: dict, patient_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    timepoints = []
    for tp_name, tp in patient["timepoints"].items():
        timepoints.append({
            "timepoint": tp_name,
            "is_day0": tp["is_day0"],
            "scan_count": len(tp["scans"]),
        })
    timepoints.sort(key=lambda t: (not t["is_day0"], t["timepoint"]))
    return timepoints


def discover_scans_in_timepoint(tree: dict, patient_name: str, timepoint_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    tp = patient["timepoints"].get(timepoint_name)
    if not tp:
        return []
    scans = []
    for scan_id, scan in sorted(tp["scans"].items()):
        scans.append({"scan_id": scan_id, "has_polygons": scan["json"] is not None})
    return scans


def all_scans_for_patient(tree: dict, patient_name: str):
    patient = tree.get(patient_name)
    if not patient:
        return []
    out = []
    for tp_name, tp in patient["timepoints"].items():
        for scan_id, scan in tp["scans"].items():
            out.append({
                "timepoint": tp_name,
                "is_day0": tp["is_day0"],
                "scan_id": scan_id,
                "has_polygons": scan["json"] is not None,
            })
    out.sort(key=lambda s: (not s["is_day0"], s["timepoint"], s["scan_id"]))
    return out


# ---------------------------------------------------------------------------
# Polygon JSON schema translation (unchanged from the local version)
# ---------------------------------------------------------------------------

def normalize_polygons(raw: dict, scan_id: str, image_size):
    regions_raw = raw.get("regions") or raw.get("polygons") or []
    regions = []
    for i, r in enumerate(regions_raw):
        poly = r.get("polygon") or r.get("points") or r.get("coords") or []
        regions.append({
            "id": r.get("id", i + 1),
            "label": r.get("label", f"region_{i + 1}"),
            "source": r.get("source", "sam2"),
            "confidence": r.get("confidence"),
            "polygon": poly,
        })
    return {
        "scan_id": raw.get("scan_id", scan_id),
        "image_size": raw.get("image_size", list(image_size)),
        "regions": regions,
    }


def load_polygons(scan: dict, scan_id: str, image_size):
    json_file = scan.get("json")
    if not json_file:
        return {"scan_id": scan_id, "image_size": list(image_size), "regions": []}
    raw = json.loads(download_file_bytes(json_file["id"]))
    return normalize_polygons(raw, scan_id, image_size)


def save_polygons(scan: dict, scan_id: str, payload: dict):
    """Write payload to Drive, backing up an original all-SAM2 file once."""
    json_file = scan.get("json")

    if json_file:
        try:
            existing = json.loads(download_file_bytes(json_file["id"]))
            regions = existing.get("regions") or existing.get("polygons") or []
            if regions and all(r.get("source", "sam2") == "sam2" for r in regions):
                backup_name = f"{scan_id}_burn_polygons.sam2_backup.json"
                siblings = _list_children(scan["id"])
                if not any(f["name"] == backup_name for f in siblings):
                    copy_file(json_file["id"], backup_name, scan["id"])
        except Exception:
            pass
        upload_json(json_file["id"], None, None, payload)
    else:
        upload_json(None, scan["id"], f"{scan_id}_burn_polygons.json", payload)

    invalidate()
