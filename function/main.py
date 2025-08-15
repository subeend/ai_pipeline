# function/main.py
import os
from datetime import datetime, timezone, timedelta
from google.cloud import storage
from google.api_core.exceptions import NotFound, PreconditionFailed
from cloudevents.http import CloudEvent
import functions_framework

# ── ENV ─────────────────────────────────────────────────────────────
PROJECT = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
REGION  = os.getenv("REGION", "asia-northeast3")

GCS_BUCKET = os.getenv("GCS_BUCKET")                     
GCS_FOLDER = os.getenv("GCS_FOLDER", "")                 
GCS_RETRAIN_FOLDER = os.getenv("GCS_RETRAIN_FOLDER", "retrain/")

MIN_IMAGES = int(os.getenv("MIN_IMAGES", "300"))

PIPE_SPEC = os.getenv("PIPE_SPEC", f"gs://{GCS_BUCKET}/pipeline/yolo_compiled.json")
OUT_MODEL_URI = os.getenv("OUT_MODEL_URI", f"gs://{GCS_BUCKET}/model/best.pt")
BASE_WEIGHTS  = os.environ["BASE_WEIGHTS"]               
RUNTIME_SA    = os.getenv("RUNTIME_SA")                  

LOCK_BLOB = os.getenv("LOCK_BLOB", f"{GCS_RETRAIN_FOLDER}.lock")
LOCK_TTL  = int(os.getenv("LOCK_TTL_HOURS", "12"))

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# ── GCS client ─────────────────────────────────────────────────────
_storage = None
def storage_client():
    global _storage
    if _storage is None:
        _storage = storage.Client()
    return _storage

def _now_utc():
    return datetime.now(timezone.utc)

# ── 쿨다운 확인 ─────────────────────────────────────────────────────
def _in_cooldown():
    b = storage_client().bucket(GCS_BUCKET).blob(LOCK_BLOB)
    if not b.exists():
        return False
    ts = (b.metadata or {}).get("ts")
    if not ts:
        return False
    try:
        old = datetime.fromisoformat(ts)
        return (_now_utc() - old) < timedelta(hours=LOCK_TTL)
    except:
        return False

# ── retrain/ 미사용 이미지 개수 ────────────────────────────────────
def _count_pending_images():
    client = storage_client()
    cnt = 0
    for b in client.list_blobs(GCS_BUCKET, prefix=GCS_RETRAIN_FOLDER):
        if b.name.lower().endswith(IMG_EXTS):
            md = (b.metadata or {})
            if md.get("used") != "true":
                cnt += 1
                if cnt >= MIN_IMAGES:
                    return cnt
    return cnt

# ── 락 ─────────────────────────────────────────────────────────────
def _try_lock():
    client = storage_client()
    blob = client.bucket(GCS_BUCKET).blob(LOCK_BLOB)

    if blob.exists():
        ts = (blob.metadata or {}).get("ts")
        if ts:
            try:
                old = datetime.fromisoformat(ts)
                if _now_utc() - old < timedelta(hours=LOCK_TTL):
                    return False
            except:
                pass
    try:
        # 없을 때만 생성 
        blob.upload_from_string("lock", content_type="text/plain", if_generation_match=0)
    except PreconditionFailed:
        return False

    md = blob.metadata or {}
    md["ts"] = _now_utc().isoformat()
    blob.metadata = md
    blob.patch()
    return True

def _unlock():
    try:
        storage_client().bucket(GCS_BUCKET).blob(LOCK_BLOB).delete()
    except NotFound:
        pass

# ── Gen2 CloudEvent 핸들러 ─────────────────────────────────────────
@functions_framework.cloud_event
def gcs_event(cloud_event: CloudEvent):
    data = cloud_event.data or {}
    name   = data.get("name", "")
    bucket = data.get("bucket", "")

    if bucket != GCS_BUCKET or not name.startswith(GCS_RETRAIN_FOLDER) or not name.lower().endswith(IMG_EXTS):
        return

    if _in_cooldown():
        print("skip: cooldown active")
        return

    pending = _count_pending_images()
    if pending < MIN_IMAGES:
        print(f"skip: pending={pending} < {MIN_IMAGES}")
        return

    if not _try_lock():
        print("skip: locked")
        return

    try:
        print("submitting pipeline...")
      
        from google.cloud import aiplatform

        aiplatform.init(project=PROJECT, location=REGION, staging_bucket=f"gs://{GCS_BUCKET}")
        job = aiplatform.PipelineJob(
            display_name="yolo-autolabel-train",
            template_path=PIPE_SPEC,
            pipeline_root=f"gs://{GCS_BUCKET}/pipeline_root",
            parameter_values={
                "bucket": GCS_BUCKET,
                "retrain_prefix": GCS_RETRAIN_FOLDER,
                "base_weights_gcs": BASE_WEIGHTS,
                "out_model_uri": OUT_MODEL_URI,
            },
        )
        job.run(service_account=RUNTIME_SA, sync=False)
        print(f"submitted with pending={pending}")
        return
    except Exception:
        _unlock()  # 제출 실패 시에만 언락
        raise
       
