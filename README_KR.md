# DEIMv2-FastTracker-TensorRT

*다른 언어로 보기: [English](README.md)*

[DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) 객체 검출기(TensorRT 가속)와
[FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker) 다중 객체 추적기(C++)를
하나로 결합한 검출·추적 파이프라인입니다.

- **검출**: DEIMv2를 TensorRT 엔진으로 변환해 추론합니다.
- **추적**: ByteTrack 계열의 FastTracker C++ 트래커를 pybind11로 묶어 Python에서 호출합니다 (OpenCV 의존성 제거, Eigen + pybind11만 필요).
- **클래스별 confidence**: class id 별로 서로 다른 confidence threshold를 최종 출력 직전에 적용합니다.
- **시각화**: 파이프라인과 분리된 함수로 박스/클래스/track id를 그립니다.

```
RGB 이미지 → 전처리 → DEIMv2 TensorRT 추론 → (옵션) FastTracker → 클래스별 confidence → 출력 → 시각화
```

## 디렉터리 구조

```
DEIMv2-FastTracker-TensorRT/
├── main.py            # 파이프라인 클래스 + 시각화 함수 + main()
├── pth2onnx.py        # DEIMv2 .pth -> .onnx
├── onnx2trt.py        # .onnx -> .engine (TensorRT)
├── requirements.txt
└── FastTracker/       # C++ 트래커 + pybind11 바인딩
    ├── include/  FastTracker.h, STrack.h, kalmanFilter.h, dataType.h, lapjv.h
    ├── src/      FastTracker.cpp, STrack.cpp, kalmanFilter.cpp, lapjv.cpp, utils.cpp
    ├── bindings.cpp
    ├── setup.py
    └── README.md      # 트래커 빌드/사용법 상세
```

## 설치

요구 사항: Python ≥ 3.7, C++14 컴파일러, CUDA + TensorRT, Eigen3.

```bash
# 1) Eigen 헤더 (C++ 트래커 빌드에 필요)
sudo apt install libeigen3-dev

# 2) Python 패키지
pip install -r requirements.txt

# 3) TensorRT 설치 (CUDA/플랫폼에 맞게 별도 설치)
pip install tensorrt

# 4) C++ FastTracker 모듈 빌드
cd FastTracker
python setup.py build_ext --inplace   # 폴더 안에 fasttracker*.so 생성
cd ..
# 또는 환경에 설치: pip install ./FastTracker
```

Eigen이 표준 경로에 없으면 `EIGEN_INCLUDE_DIR` 환경변수로 `Eigen/` 폴더가 있는 경로를 지정하세요.

## 사용법

### 1. 모델 변환 (.pth → .onnx → .engine)

ONNX 내보내기는 **DEIMv2 레포 내부**에서 실행해야 합니다 (`engine.core.YAMLConfig`, `.deploy()` 필요).
`pth2onnx.py`를 DEIMv2 레포 루트로 복사한 뒤 실행하세요.

```bash
# (DEIMv2 레포 안에서) pth -> onnx
python pth2onnx.py -c configs/deimv2/deimv2_dinov3_s_coco.yml -r deimv2_s.pth --check --simplify

# onnx -> tensorrt engine
python3 onnx2trt.py --onnx deimv2_s.onnx --saveEngine deimv2_s.engine --fp16 --size 640
```

### 2. 파이프라인 실행

입력 폴더의 모든 이미지를 읽어 검출/추적 후 시각화 결과를 출력 폴더에 저장합니다.

```bash
# 검출 + 추적
python3 main.py \
    --input ./images --output ./results \
    --trt ./deimv2_s.engine --size 640 --model-size s \
    --track --conf 0.4 0.5 0.3 \
    --names person,bike,car

# 검출만 (추적 비활성화)
python3 main.py -i ./images -o ./results -trt ./deimv2_s.engine -s 640 -ms s --conf 0.4 0.5 0.3
```

#### 주요 인자

| 인자 | 설명 |
|------|------|
| `-i, --input` | 입력 이미지 폴더 |
| `-o, --output` | 결과 저장 폴더 |
| `-trt, --trt` | TensorRT 엔진(`.engine`) 경로 |
| `--track` | FastTracker 추적 활성화 (생략 시 검출만) |
| `--conf` | 클래스별 confidence threshold 리스트 (인덱스 = class id) |
| `-s, --size` | 모델 입력 크기 (예: 640) |
| `-ms, --model-size` | `atto/femto/pico/n/s/m/l/x` (정규화 적용 여부 결정) |
| `--names` | 클래스 이름 (콤마 구분 문자열 또는 파일, 1줄당 1개) |
| `-d, --device` | 추론 디바이스 (기본 `cuda:0`) |
| `--track-buffer` | lost track 유지 프레임 수 (기본 30) |
| `--frame-rate` | 입력 프레임레이트 (기본 30) |

> `--conf`는 인덱스가 class id에 대응합니다. 예: `--conf 0.4 0.5 0.3` → class 0은 0.4, class 1은 0.5, class 2는 0.3.
> 리스트에 없는 class id는 마지막 값을 사용합니다. 추적 사용 시에도 confidence는 추적 **이후**, 최종 출력 직전에 적용됩니다.

## Python API

`main.py`의 `DEIMv2FastTracker` 클래스를 직접 사용할 수도 있습니다.

```python
import cv2
from main import DEIMv2FastTracker, visualize

pipe = DEIMv2FastTracker(
    engine_path="deimv2_s.engine",
    input_size=640,
    model_size="s",
    device="cuda:0",
)

bgr = cv2.imread("frame.jpg")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# (K, 7) float32: [x1, y1, x2, y2, score, class_id, track_id]
out = pipe.run(rgb, track=True, class_conf=[0.4, 0.5, 0.3])

vis = visualize(rgb, out, names=["person", "bike", "car"])  # 저장용 BGR 이미지
cv2.imwrite("result.jpg", vis)
```

### 메서드

- `preprocess(rgb_image)` — RGB numpy 이미지를 엔진 입력 blob으로 변환
- `infer(blob)` — TensorRT 추론, `(labels, boxes, scores)` 반환
- `track(dets)` — `(N,6)` 검출을 C++ FastTracker에 입력, `(M,7)` 반환
- `apply_class_conf(arr, class_conf)` — 클래스별 confidence 필터
- `run(rgb_image, track=True, class_conf=None)` — 위 단계를 한 번에 실행

추적은 클래스 인스턴스 내부에서 프레임 간 상태를 유지합니다. 폴더(시퀀스) 단위로 새 `DEIMv2FastTracker`를 만들면 track id가 새로 시작됩니다.

## 출력 포맷

`run()`이 반환하는 배열의 각 행은 다음과 같습니다.

```
[x1, y1, x2, y2, score, class_id, track_id]
```

- 좌표는 원본 이미지 기준의 `xyxy`입니다.
- `track_id`는 추적 비활성화(`track=False`) 시 `-1`입니다.

## 라이선스 / 출처

- DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2
- FastTracker: https://github.com/Hamidreza-Hashempoor/FastTracker
