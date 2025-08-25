from kfp import dsl
from kfp.dsl import component, OutputPath

PY_BASE = "us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-0:latest"

# ---------- AutoLabel ----------
@component(
    base_image=PY_BASE,
    packages_to_install=[
        "ultralytics==8.3.0",
        "google-cloud-storage>=2.16.0",
        "opencv-python-headless",
        "numpy",
        "pyyaml",
    ],
)
def autolabel_step(
    bucket: str,
    retrain_prefix: str,
    base_weights_gcs: str,
    image_exts: str = ".png,.jpg,.jpeg,.bmp,.tif,.tiff",
    score_thresh: float = 0.25,
):
    """gs://{bucket}/{retrain_prefix}/ 아래 이미지에 YOLO 추론 → 같은 경로 .txt 저장(이미 있으면 스킵)."""
    import os
    import cv2, numpy as np
    from ultralytics import YOLO
    from google.cloud import storage

    exts = tuple([e.strip().lower() for e in image_exts.split(",") if e.strip()])
    gcs = storage.Client()
    bkt = gcs.bucket(bucket)

    # 가중치 다운로드
    def gcs_to_local(uri: str, local_path: str):
        assert uri.startswith("gs://")
        b, p = uri[5:].split("/", 1)
        storage.Client().bucket(b).blob(p).download_to_filename(local_path)

    local_w = "/tmp/base.pt"
    gcs_to_local(base_weights_gcs, local_w)
    model = YOLO(local_w)

    for blob in gcs.list_blobs(bucket, prefix=retrain_prefix):
        name = blob.name
        if not name.lower().endswith(exts):
            continue
        txt_name = os.path.splitext(name)[0] + ".txt"
        if bkt.blob(txt_name).exists():
            continue

        img_bytes = blob.download_as_bytes()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

        results = model.predict(source=img, conf=score_thresh, verbose=False)
        lines = []
        for r in results:
            h, w = r.orig_shape
            for b in r.boxes:
                cls = int(b.cls[0]) if b.cls is not None else 0
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cx = ((x1 + x2) / 2.0) / w
                cy = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        bkt.blob(txt_name).upload_from_string("\n".join(lines) if lines else "", content_type="text/plain")


# ---------- Train + Gate (+ Registry/Versioning) ----------
@component(
    base_image=PY_BASE,
    packages_to_install=[
        "ultralytics==8.3.0",
        "google-cloud-storage>=2.16.0",
        "google-cloud-aiplatform>=1.50.0",
        "pyyaml",
    ],
)
def train_step(
    bucket: str,
    retrain_prefix: str,
    base_weights_gcs: str,
    out_model_uri: str,
    accepted: OutputPath[str],        
    epochs: int = 30,
    imgsz: int = 640,
    min_map50: float = 0.60,
    min_improve: float = 0.00,
):
    """
    retrain_prefix 데이터로 재학습 → 베이스대비 mAP50 비교.
    - 항상: 버전 폴더(models/yolo/versions/{ts}/)에 best.pt 및 metrics.json 저장
    - 가능하면: Vertex AI Model Registry에 업로드(+ alias)
    - 게이트 통과 시에만: out_model_uri(GCS prod 포인터) 갱신 및 포인터 메타데이터에 version_path 기록
    """
    import os, csv, yaml, json
    from datetime import datetime, timezone
    from ultralytics import YOLO
    from google.cloud import storage
    from google.cloud import aiplatform as aip

    IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    gcs = storage.Client()
    bkt = gcs.bucket(bucket)

    # -------- 데이터셋 구성 --------
    ds = "/tmp/ds"; os.makedirs(ds, exist_ok=True)
    img_dir = os.path.join(ds, "images"); lbl_dir = os.path.join(ds, "labels")
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)

    for blob in gcs.list_blobs(bucket, prefix=retrain_prefix):
        if blob.name.lower().endswith(IMG_EXTS):
            img_local = os.path.join(img_dir, os.path.basename(blob.name))
            blob.download_to_filename(img_local)
            txt_name = os.path.splitext(blob.name)[0] + ".txt"
            lbl_local = os.path.join(lbl_dir, os.path.basename(txt_name))
            tblob = bkt.blob(txt_name)
            if tblob.exists():
                tblob.download_to_filename(lbl_local)
            else:
                open(lbl_local, "w").close()  # negative label

    data_yaml = os.path.join(ds, "data.yaml")
    with open(data_yaml, "w") as f:
        yaml.safe_dump({"path": ds, "train": "images", "val": "images", "names": ["defect"]}, f)

    # -------- mAP50 파서 --------
    def parse_last_map50(csv_path: str) -> float:
        if not os.path.exists(csv_path): return float("nan")
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows: return float("nan")
        last = rows[-1]
        for k in last.keys():
            lk = k.lower()
            if "map50" in lk and "95" not in lk:
                try: return float(last[k])
                except: pass
        return float("nan")

    # -------- 베이스 성능 --------
    local_w = "/tmp/base.pt"
    wb, wo = base_weights_gcs[5:].split("/", 1)
    storage.Client().bucket(wb).blob(wo).download_to_filename(local_w)

    base_model = YOLO(local_w)
    base_proj = "/tmp/run_base"
    base_model.val(data=data_yaml, imgsz=imgsz, project=base_proj, name="val", verbose=False)
    base_map50 = parse_last_map50(os.path.join(base_proj, "val", "results.csv"))

    # -------- 학습 --------
    model = YOLO(local_w)
    new_proj = "/tmp/run"
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, project=new_proj, name="train", exist_ok=True)
    best_local = os.path.join(new_proj, "train", "weights", "best.pt")

    # -------- 새 성능 --------
    new_model = YOLO(best_local)
    new_proj_val = "/tmp/run_new"
    new_model.val(data=data_yaml, imgsz=imgsz, project=new_proj_val, name="val", verbose=False)
    new_map50 = parse_last_map50(os.path.join(new_proj_val, "val", "results.csv"))

    print(f"[gate] base mAP50={base_map50:.4f}, new mAP50={new_map50:.4f}, "
          f"min_map50={min_map50}, min_improve={min_improve}")

    passed = False
    try:
        if (new_map50 >= min_map50) and ((new_map50 - base_map50) >= min_improve):
            passed = True
    except:
        passed = False

    # -------- 버전 폴더 저장(항상) --------
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    version_prefix = f"models/yolo/versions/{ts}/"
    bkt.blob(version_prefix + "best.pt").upload_from_filename(best_local)
    metrics = {
        "base_map50": base_map50,
        "new_map50": new_map50,
        "delta_map50": (new_map50 - base_map50) if (not (base_map50!=base_map50 or new_map50!=new_map50)) else None,
        "epochs": epochs, "imgsz": imgsz,
        "retrain_prefix": retrain_prefix,
        "timestamp": ts,
    }
    bkt.blob(version_prefix + "metrics.json").upload_from_string(json.dumps(metrics, indent=2), content_type="application/json")

    # -------- Model Registry 등록--------
    try:
        proj = os.getenv("AIP_PROJECT_NUMBER") or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
        loc  = os.getenv("AIP_LOCATION") or os.getenv("REGION", "asia-northeast3")
        aip.init(project=proj, location=loc)
        model_obj = aip.Model.upload(
            display_name="yolo-detector",
            artifact_uri=f"gs://{bucket}/{version_prefix}",
            labels={"source":"pipeline","ts":ts},
        )
        try:
            model_obj.upload_model_evaluation(
                metrics={"map50": new_map50, "base_map50": base_map50},
                display_name=f"eval_{ts}",
            )
        except Exception as e:
            print("eval upload skipped:", e)
        try:
            model_obj.add_version_aliases(["prod" if passed else "staging"])
        except Exception as e:
            print("alias skipped:", e)
        print("model_registry_id:", model_obj.resource_name)
    except Exception as e:
        print("model registry upload skipped:", e)

    # -------- 게이트 통과 시에만 prod 포인터(out_model_uri) 교체 --------
    if passed:
        out_bkt, out_blob = out_model_uri[5:].split("/", 1)
        storage.Client().bucket(out_bkt).blob(out_blob).upload_from_filename(best_local)
        # 포인터 오브젝트에 원본 버전 경로 메타데이터 기록
        ob = storage.Client().bucket(out_bkt).blob(out_blob)
        md = (ob.metadata or {})
        md.update({"version_path": f"gs://{bucket}/{version_prefix}best.pt", "updated_at": ts})
        ob.metadata = md
        ob.patch()
        print("[train] uploaded best.pt to", out_model_uri)
    else:
        print("[train] gate failed → NOT uploading to", out_model_uri)

    # 게이트 결과 출력
    with open(accepted, "w") as f:
        f.write("true" if passed else "false")


# ---------- Mark Used ----------
@component(
    base_image=PY_BASE,
    packages_to_install=["google-cloud-storage>=2.16.0"],
)
def mark_used_step(bucket: str, retrain_prefix: str):
    """게이트 통과 시에만 사용된 이미지에 used=true 메타데이터 세팅."""
    from google.cloud import storage
    IMG_EXTS = (".png",".jpg",".jpeg",".bmp",".tif",".tiff")
    gcs = storage.Client()
    bkt = gcs.bucket(bucket)
    for blob in gcs.list_blobs(bucket, prefix=retrain_prefix):
        if not blob.name.lower().endswith(IMG_EXTS):
            continue
        b = bkt.blob(blob.name)
        md = b.metadata or {}
        if md.get("used") == "true":
            continue
        md["used"] = "true"
        b.metadata = md
        b.patch()


# ---------- Pipeline DAG ----------
@dsl.pipeline(name="yolo-autolabel-train")
def yolo_pipeline(
    bucket: str,
    retrain_prefix: str,
    base_weights_gcs: str,
    out_model_uri: str,
    score_thresh: float = 0.25,
    epochs: int = 30,
    imgsz: int = 640,
    min_map50: float = 0.60,
    min_improve: float = 0.00,
):
    a = autolabel_step(bucket=bucket, retrain_prefix=retrain_prefix,
                       base_weights_gcs=base_weights_gcs, score_thresh=score_thresh)

    t = train_step(bucket=bucket, retrain_prefix=retrain_prefix,
                   base_weights_gcs=base_weights_gcs, out_model_uri=out_model_uri,
                   epochs=epochs, imgsz=imgsz,
                   min_map50=min_map50, min_improve=min_improve)
    t.after(a)

   
    with dsl.If(t.outputs["accepted"] == "true"):
        m = mark_used_step(bucket=bucket, retrain_prefix=retrain_prefix)
        m.after(t)
