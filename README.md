# ai_pipeline

Cloud Run 기반 **자동 재학습 트리거 서비스**와 Vertex AI **파이프라인 스펙**을 함께 관리하는 모노리포입니다.

- `function/` — Cloud Run에 배포되는 HTTP 서비스(트리거/헬스체크).  
  GCS 폴더의 이미지 개수를 확인하고, 임계치(예: 300장) 이상이면 **Vertex AI Custom Job / Pipeline** 실행을 트리거합니다.
- `pipeline/` — Vertex AI(KFP) **컴파일된 파이프라인 스펙**과 관련 스크립트/노트북을 둡니다.  

---

## 폴더 구조

ai_pipeline/
├─ function/                 # Cloud Run에 배포되는 HTTP 트리거/헬스체크 서비스
│  ├─ main.py               # 엔트리포인트(예시)
│  └─ requirements.txt      # 파이썬 의존성
├─ pipeline/                # Vertex AI/KFP 파이프라인 스펙 및 도구
│  ├─ yolo_compiled.json    # 컴파일된 파이프라인 스펙(등록만 하고 실행 금지)
│  ├─ pipeline.py           # KFP v2 파이프라인 정의
│  └─ compile.py         # 파이프라인 스펙(JSON)으로 컴파일
├─ .gitignore               # 공통 무시 규칙
└─ README.md                # 리포 개요(이 파일)

