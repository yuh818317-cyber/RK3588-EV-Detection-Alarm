import argparse
import csv
import json
import os
import time
import threading
import subprocess
import sqlite3

import yaml

import cv2
import numpy as np
from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    send_from_directory,
    request
)
from rknnlite.api import RKNNLite


cv2.setNumThreads(1)

app = Flask(__name__)

# 第 5 步先使用你上传的视频测试。
# 后面恢复实时摄像头时，把 SOURCE_MODE 改成 "camera"。
SOURCE_MODE = "video"
CAMERA_DEVICE = "/dev/video21"
VIDEO_PATH = "/root/ev_camera_project/roi_test.mp4"
LOOP_VIDEO = True
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 10

MODEL_PATH = "/root/ev_camera_project/models/yolov8n_rk3588.rknn"
ROI_CONFIG_PATH = "/root/ev_camera_project/config/roi_config.json"
SAVE_DIR = "/root/ev_camera_project/snapshots"
LOG_DIR = "/root/ev_camera_project/logs"
EVENT_LOG_PATH = os.path.join(LOG_DIR, "event_log.csv")

# 第 9 步：截图留证和 SQLite 本地事件数据库
DATABASE_PATH = "/root/ev_camera_project/data/events.db"
EVIDENCE_DIR = "/root/ev_camera_project/evidence/screenshots"
SCENE_NAME = "building_entrance"
RISK_LEVEL = "high"

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
DETECT_PERIOD = 0.5

# 第 6 步：入口线与方向判断参数
# 入口线单独设置，不再绑定 ROI 的第一条边。
# 坐标采用 0 到 1 的比例，适配不同分辨率。
ENTRY_LINE_POINTS = [
    [0.47, 0.29],
    [0.82, 0.382]
]
TRACK_MAX_DISTANCE = 140
TRACK_MAX_MISSED = 4
CONFIRM_MISS_TOLERANCE = 2
DETECTION_HOLD_CYCLES = 2
POSITION_HISTORY_LENGTH = 12

# 第 7 步：连续多帧确认和全局报警冷却
CONFIRM_HIT_FRAMES = 3
EVENT_COOLDOWN_SECONDS = 10.0
EVENT_TYPE = "electric_vehicle_enter"

# 第 8 步：本地 LED 报警参数
# 当前使用 P26 40Pin 排针中的 GPIO3_B5。
# Linux GPIO 对应 gpiochip3 的 line 13。
ALARM_MODE = "led"       # 可改成 "fake"，在不接硬件时模拟报警
LED_GPIO_CHIP = "gpiochip3"
LED_GPIO_LINE = 13
LED_ALARM_DURATION_SECONDS = 5

WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
WEB_THREADED = True

# 性能档位：smooth / balanced / quality / custom
# smooth：优先流畅；balanced：比赛推荐；quality：优先网页清晰度。
PERFORMANCE_MODE = "balanced"
WEB_STREAM_FPS = 15.0
JPEG_QUALITY = 75
PERFORMANCE_LOG_INTERVAL = 1.0

PERFORMANCE_PRESETS = {
    "smooth": {
        "camera_fps": 20.0,
        "detect_period": 0.12,
        "web_stream_fps": 20.0,
        "jpeg_quality": 65
    },
    "balanced": {
        "camera_fps": 30.0,
        "detect_period": 0.15,
        "web_stream_fps": 12.0,
        "jpeg_quality": 70
    },
    "quality": {
        "camera_fps": 15.0,
        "detect_period": 0.12,
        "web_stream_fps": 12.0,
        "jpeg_quality": 82
    }
}

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

TARGET_CLASSES = {"bicycle", "motorcycle"}

DEFAULT_ROI_POINTS = [
    [0.20, 0.45],
    [0.80, 0.45],
    [0.95, 0.95],
    [0.05, 0.95]
]


def parse_command_line():
    """读取主程序启动参数。"""
    parser = argparse.ArgumentParser(
        description="RK3588 电动车违规入内检测主程序"
    )
    parser.add_argument(
        "--config",
        default="configs/dorm_gate.yaml",
        help="YAML 主配置文件路径"
    )
    return parser.parse_args()


def get_config_value(config, key_path, default):
    """按 input.mode 这样的层级路径读取配置。"""
    current = config

    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current


def config_bool(value, default=False):
    """把 YAML 中的布尔值安全转换成 Python bool。"""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "on"}:
            return True
        if normalized in {"false", "no", "0", "off"}:
            return False

    if value is None:
        return default

    return bool(value)


def resolve_project_path(value, project_root):
    """绝对路径直接使用，相对路径相对于项目根目录。"""
    path = os.path.expanduser(str(value))

    if os.path.isabs(path):
        return os.path.normpath(path)

    return os.path.normpath(os.path.join(project_root, path))


ARGS = parse_command_line()
MAIN_CONFIG_PATH = os.path.abspath(
    os.path.expanduser(ARGS.config)
)

if not os.path.isfile(MAIN_CONFIG_PATH):
    raise FileNotFoundError(
        f"主配置文件不存在：{MAIN_CONFIG_PATH}"
    )

with open(MAIN_CONFIG_PATH, "r", encoding="utf-8") as file:
    MAIN_CONFIG = yaml.safe_load(file) or {}

if not isinstance(MAIN_CONFIG, dict):
    raise ValueError("YAML 主配置的最外层必须是键值结构")

PROJECT_ROOT = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "project.root",
        "/root/ev_camera_project"
    ),
    os.getcwd()
)

SOURCE_MODE = str(
    get_config_value(MAIN_CONFIG, "input.mode", SOURCE_MODE)
).strip().lower()
CAMERA_DEVICE = str(
    get_config_value(
        MAIN_CONFIG,
        "input.camera_device",
        CAMERA_DEVICE
    )
)
VIDEO_PATH = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "input.video_path",
        VIDEO_PATH
    ),
    PROJECT_ROOT
)
LOOP_VIDEO = config_bool(
    get_config_value(
        MAIN_CONFIG,
        "input.loop_video",
        LOOP_VIDEO
    ),
    LOOP_VIDEO
)
CAMERA_WIDTH = int(
    get_config_value(
        MAIN_CONFIG,
        "input.camera_width",
        CAMERA_WIDTH
    )
)
CAMERA_HEIGHT = int(
    get_config_value(
        MAIN_CONFIG,
        "input.camera_height",
        CAMERA_HEIGHT
    )
)
CAMERA_FPS = float(
    get_config_value(
        MAIN_CONFIG,
        "input.camera_fps",
        CAMERA_FPS
    )
)

MODEL_PATH = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "model.path",
        MODEL_PATH
    ),
    PROJECT_ROOT
)
INPUT_SIZE = int(
    get_config_value(
        MAIN_CONFIG,
        "model.input_size",
        INPUT_SIZE
    )
)
CONF_THRESHOLD = float(
    get_config_value(
        MAIN_CONFIG,
        "model.conf_threshold",
        CONF_THRESHOLD
    )
)
NMS_THRESHOLD = float(
    get_config_value(
        MAIN_CONFIG,
        "model.nms_threshold",
        NMS_THRESHOLD
    )
)
TARGET_CLASSES = set(
    get_config_value(
        MAIN_CONFIG,
        "model.target_classes",
        sorted(TARGET_CLASSES)
    )
)

ROI_CONFIG_PATH = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "roi.path",
        ROI_CONFIG_PATH
    ),
    PROJECT_ROOT
)
DEFAULT_ROI_POINTS = get_config_value(
    MAIN_CONFIG,
    "roi.default_points",
    DEFAULT_ROI_POINTS
)
ENTRY_LINE_POINTS = get_config_value(
    MAIN_CONFIG,
    "entry_line.points",
    ENTRY_LINE_POINTS
)

DETECT_PERIOD = float(
    get_config_value(
        MAIN_CONFIG,
        "detection.period_seconds",
        DETECT_PERIOD
    )
)
CONFIRM_HIT_FRAMES = int(
    get_config_value(
        MAIN_CONFIG,
        "detection.confirm_hit_frames",
        CONFIRM_HIT_FRAMES
    )
)
EVENT_COOLDOWN_SECONDS = float(
    get_config_value(
        MAIN_CONFIG,
        "detection.event_cooldown_seconds",
        EVENT_COOLDOWN_SECONDS
    )
)
EVENT_TYPE = str(
    get_config_value(
        MAIN_CONFIG,
        "detection.event_type",
        EVENT_TYPE
    )
)

TRACK_MAX_DISTANCE = float(
    get_config_value(
        MAIN_CONFIG,
        "tracking.max_distance",
        TRACK_MAX_DISTANCE
    )
)
TRACK_MAX_MISSED = int(
    get_config_value(
        MAIN_CONFIG,
        "tracking.max_missed",
        TRACK_MAX_MISSED
    )
)
POSITION_HISTORY_LENGTH = int(
    get_config_value(
        MAIN_CONFIG,
        "tracking.history_length",
        POSITION_HISTORY_LENGTH
    )
)

ALARM_MODE = str(
    get_config_value(
        MAIN_CONFIG,
        "alarm.mode",
        ALARM_MODE
    )
).strip().lower()
LED_GPIO_CHIP = str(
    get_config_value(
        MAIN_CONFIG,
        "alarm.gpio_chip",
        LED_GPIO_CHIP
    )
)
LED_GPIO_LINE = int(
    get_config_value(
        MAIN_CONFIG,
        "alarm.gpio_line",
        LED_GPIO_LINE
    )
)
LED_ALARM_DURATION_SECONDS = float(
    get_config_value(
        MAIN_CONFIG,
        "alarm.duration_seconds",
        LED_ALARM_DURATION_SECONDS
    )
)

SAVE_DIR = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "storage.manual_snapshot_dir",
        SAVE_DIR
    ),
    PROJECT_ROOT
)
LOG_DIR = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "storage.log_dir",
        LOG_DIR
    ),
    PROJECT_ROOT
)
EVENT_LOG_PATH = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "storage.csv_log_path",
        os.path.join(LOG_DIR, "event_log.csv")
    ),
    PROJECT_ROOT
)
DATABASE_PATH = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "storage.database_path",
        DATABASE_PATH
    ),
    PROJECT_ROOT
)
EVIDENCE_DIR = resolve_project_path(
    get_config_value(
        MAIN_CONFIG,
        "storage.evidence_dir",
        EVIDENCE_DIR
    ),
    PROJECT_ROOT
)
SCENE_NAME = str(
    get_config_value(
        MAIN_CONFIG,
        "storage.scene_name",
        SCENE_NAME
    )
)
RISK_LEVEL = str(
    get_config_value(
        MAIN_CONFIG,
        "storage.risk_level",
        RISK_LEVEL
    )
)

WEB_HOST = str(
    get_config_value(
        MAIN_CONFIG,
        "web.host",
        WEB_HOST
    )
)
WEB_PORT = int(
    get_config_value(
        MAIN_CONFIG,
        "web.port",
        WEB_PORT
    )
)
WEB_THREADED = config_bool(
    get_config_value(
        MAIN_CONFIG,
        "web.threaded",
        WEB_THREADED
    ),
    WEB_THREADED
)

PERFORMANCE_MODE = str(
    get_config_value(
        MAIN_CONFIG,
        "performance.mode",
        PERFORMANCE_MODE
    )
).strip().lower()

if PERFORMANCE_MODE in PERFORMANCE_PRESETS:
    preset = PERFORMANCE_PRESETS[PERFORMANCE_MODE]
    CAMERA_FPS = float(preset["camera_fps"])
    DETECT_PERIOD = float(preset["detect_period"])
    WEB_STREAM_FPS = float(preset["web_stream_fps"])
    JPEG_QUALITY = int(preset["jpeg_quality"])
elif PERFORMANCE_MODE == "custom":
    WEB_STREAM_FPS = float(
        get_config_value(
            MAIN_CONFIG,
            "performance.web_stream_fps",
            WEB_STREAM_FPS
        )
    )
    JPEG_QUALITY = int(
        get_config_value(
            MAIN_CONFIG,
            "performance.jpeg_quality",
            JPEG_QUALITY
        )
    )
    PERFORMANCE_LOG_INTERVAL = float(
        get_config_value(
            MAIN_CONFIG,
            "performance.log_interval_seconds",
            PERFORMANCE_LOG_INTERVAL
        )
    )
else:
    raise ValueError(
        "performance.mode 只能填写 smooth、balanced、quality 或 custom"
    )

WEB_STREAM_FPS = max(1.0, min(WEB_STREAM_FPS, 30.0))
JPEG_QUALITY = max(40, min(JPEG_QUALITY, 95))
DETECT_PERIOD = max(0.01, DETECT_PERIOD)
PERFORMANCE_LOG_INTERVAL = max(0.2, PERFORMANCE_LOG_INTERVAL)

if SOURCE_MODE not in {"camera", "video"}:
    raise ValueError(
        "input.mode 只能填写 camera 或 video"
    )

if ALARM_MODE not in {"led", "fake"}:
    raise ValueError(
        "alarm.mode 只能填写 led 或 fake"
    )

if len(ENTRY_LINE_POINTS) != 2:
    raise ValueError("entry_line.points 必须包含两个点")

if len(DEFAULT_ROI_POINTS) < 3:
    raise ValueError("roi.default_points 至少需要三个点")

print(f"[CONFIG] 主配置已加载：{MAIN_CONFIG_PATH}")
print(f"[CONFIG] 项目根目录：{PROJECT_ROOT}")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ROI_CONFIG_PATH), exist_ok=True)
os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

frame_lock = threading.Lock()
# ROI 和入口线共用同一把可重入锁，保证网页保存与检测线程读取不会冲突。
roi_lock = threading.RLock()
tracking_lock = threading.RLock()
database_lock = threading.Lock()

latest_raw_frame = None
latest_display_frame = None
latest_detections = []

latest_preprocess_ms = 0.0
latest_inference_ms = 0.0
latest_postprocess_ms = 0.0
latest_display_fps = 0.0

roi_points_normalized = [list(point) for point in DEFAULT_ROI_POINTS]
entry_line_points_normalized = [list(point) for point in ENTRY_LINE_POINTS]
roi_config_mtime = None

running = True

# 简易参考点跟踪器状态
tracks = {}
next_track_id = 1
event_count = 0
latest_event_text = "Waiting for crossing event"
latest_event_time = 0.0
last_alarm_time = 0.0


class FakeAlarm:
    """不接硬件时使用的模拟报警器。"""

    def __init__(self):
        print("[FakeAlarm] 模拟报警器初始化完成")

    def trigger(self, reason="检测到违规电动车"):
        print(f"[FakeAlarm] 模拟报警：{reason}")
        return True

    def close(self):
        print("[FakeAlarm] 已安全退出")


class LEDAlarm:
    """
    使用 libgpiod 的 gpioset 命令控制 LED。

    GPIO 输出 1：LED 点亮
    GPIO 输出 0：LED 熄灭
    """

    def __init__(
        self,
        chip="gpiochip3",
        line=13,
        alarm_duration=1
    ):
        self.chip = chip
        self.line = line
        self.alarm_duration = max(1, int(round(alarm_duration)))

        self.lock = threading.Lock()
        self.process = None
        self.worker = None
        self.closed = False

        self.off()
        print(
            f"[LEDAlarm] 初始化完成："
            f"{self.chip} line {self.line}"
        )

    def _run_level(self, level, seconds):
        """让 GPIO 保持指定电平一段时间。"""
        process = subprocess.Popen(
            [
                "gpioset",
                "-m", "time",
                "-s", str(max(1, int(seconds))),
                self.chip,
                f"{self.line}={level}"
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        with self.lock:
            self.process = process

        _, stderr_text = process.communicate()

        with self.lock:
            if self.process is process:
                self.process = None

        if process.returncode not in (0, -15):
            message = (stderr_text or "").strip()
            raise RuntimeError(
                f"gpioset 执行失败，返回码 {process.returncode}："
                f"{message}"
            )

    def _alarm_worker(self, reason):
        try:
            print("========== LED ALARM ON ==========")
            print(f"报警原因：{reason}")

            self._run_level(
                level=1,
                seconds=self.alarm_duration
            )

        except Exception as error:
            print(f"[LEDAlarm] 点亮失败：{error}")

        finally:
            try:
                self._run_level(level=0, seconds=1)
            except Exception as error:
                print(f"[LEDAlarm] 熄灭失败：{error}")

            print("========== LED ALARM OFF =========")

    def trigger(self, reason="检测到违规电动车"):
        """
        非阻塞触发 LED 报警。

        主程序中的 EVENT_COOLDOWN_SECONDS 负责限制重复报警。
        """
        with self.lock:
            if self.closed:
                print("[LEDAlarm] 已关闭，忽略本次触发")
                return False

            if self.worker is not None and self.worker.is_alive():
                print("[LEDAlarm] LED 当前正在报警，忽略重复触发")
                return False

            self.worker = threading.Thread(
                target=self._alarm_worker,
                args=(reason,),
                daemon=True
            )
            self.worker.start()

        return True

    def off(self):
        """立即终止当前 gpioset 进程，并强制输出低电平。"""
        with self.lock:
            process = self.process

        if process is not None and process.poll() is None:
            process.terminate()

            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        subprocess.run(
            [
                "gpioset",
                "-m", "time",
                "-s", "1",
                self.chip,
                f"{self.line}=0"
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        with self.lock:
            if self.process is process:
                self.process = None

    def close(self):
        """程序退出时保证 LED 熄灭。"""
        with self.lock:
            self.closed = True

        self.off()
        print("[LEDAlarm] 已关闭，LED 保持熄灭")


def reset_tracking_state():
    """视频循环、输入切换或规则调整时安全清空旧目标。"""
    global tracks
    global next_track_id

    with tracking_lock:
        tracks = {}
        next_track_id = 1

    print("目标跟踪状态已重置")


def ensure_event_log():
    """确保事件日志文件存在，并写入表头。"""
    if os.path.exists(EVENT_LOG_PATH) and os.path.getsize(EVENT_LOG_PATH) > 0:
        return

    with open(
        EVENT_LOG_PATH,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "alarm_start_time",
            "event_type",
            "track_id",
            "class_name",
            "confidence"
        ])

    print(f"已创建事件日志：{EVENT_LOG_PATH}")


def append_event_log(
    timestamp,
    event_type,
    track_id,
    class_name,
    confidence
):
    """把一次正式报警追加到 CSV 日志中。"""
    ensure_event_log()

    time_text = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp)
    )

    with open(
        EVENT_LOG_PATH,
        "a",
        encoding="utf-8",
        newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            time_text,
            event_type,
            track_id,
            class_name,
            f"{confidence:.3f}"
        ])



def ensure_event_database():
    """创建第 9 步使用的 SQLite 数据库和事件表。"""
    with database_lock:
        connection = sqlite3.connect(DATABASE_PATH)

        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time TEXT NOT NULL,
                    scene TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    screenshot_path TEXT,
                    track_id INTEGER,
                    class_name TEXT
                )
                """
            )
            connection.commit()
        finally:
            connection.close()

    print(f"SQLite 事件数据库已就绪：{DATABASE_PATH}")


def build_evidence_frame(frame, detection, track_id, timestamp):
    """在报警原始画面上补充关键标记，生成更直观的留证截图。"""
    evidence = frame.copy()
    polygon = get_roi_polygon(evidence.shape)
    line_start, line_end = get_entry_line(evidence.shape)

    cv2.polylines(
        evidence,
        [polygon],
        True,
        (0, 165, 255),
        3
    )
    cv2.line(
        evidence,
        line_start,
        line_end,
        (255, 255, 0),
        4
    )

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    center_x = int(detection["center_x"])
    center_y = int(detection["center_y"])

    cv2.rectangle(
        evidence,
        (x1, y1),
        (x2, y2),
        (0, 0, 255),
        3
    )
    cv2.circle(
        evidence,
        (center_x, center_y),
        7,
        (0, 0, 255),
        -1
    )

    label = (
        f"ALARM ID {track_id} "
        f"{detection['class_name']} "
        f"{detection['score']:.2f}"
    )
    cv2.putText(
        evidence,
        label,
        (x1, max(30, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2
    )

    time_text = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp)
    )
    cv2.putText(
        evidence,
        time_text,
        (20, evidence.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return evidence


def save_event_screenshot(frame, detection, track_id, timestamp):
    """自动保存一次报警事件截图，文件名包含日期和毫秒。"""
    if frame is None:
        raise RuntimeError("报警时没有可用画面")

    milliseconds = int((timestamp % 1) * 1000)
    filename = time.strftime(
        "event_%Y%m%d_%H%M%S",
        time.localtime(timestamp)
    )
    filename += f"_{milliseconds:03d}_id{track_id}.jpg"
    filepath = os.path.join(EVIDENCE_DIR, filename)

    evidence_frame = build_evidence_frame(
        frame=frame,
        detection=detection,
        track_id=track_id,
        timestamp=timestamp
    )

    if not cv2.imwrite(filepath, evidence_frame):
        raise RuntimeError(f"截图写入失败：{filepath}")

    print(f"报警截图已保存：{filepath}")
    return filepath


def insert_event_database(
    timestamp,
    scene,
    event_type,
    risk_level,
    confidence,
    screenshot_path,
    track_id,
    class_name
):
    """把一次报警事件写入 SQLite。"""
    event_time = time.strftime(
        "%Y-%m-%d %H:%M:%S",
        time.localtime(timestamp)
    )

    with database_lock:
        connection = sqlite3.connect(DATABASE_PATH)

        try:
            cursor = connection.execute(
                """
                INSERT INTO events (
                    event_time,
                    scene,
                    event_type,
                    risk_level,
                    confidence,
                    screenshot_path,
                    track_id,
                    class_name
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_time,
                    scene,
                    event_type,
                    risk_level,
                    float(confidence),
                    screenshot_path,
                    int(track_id),
                    class_name
                )
            )
            connection.commit()
            event_id = cursor.lastrowid
        finally:
            connection.close()

    print(f"SQLite 事件记录已写入，事件编号：{event_id}")
    return event_id


def save_event_record(frame, detection, track_id, timestamp):
    """
    完成一次完整留证。

    截图失败时仍写入数据库，并把截图路径留空，
    避免留证功能异常导致主检测程序退出。
    """
    screenshot_path = ""

    try:
        screenshot_path = save_event_screenshot(
            frame=frame,
            detection=detection,
            track_id=track_id,
            timestamp=timestamp
        )
    except Exception as error:
        print(f"报警截图保存失败：{error}")

    try:
        return insert_event_database(
            timestamp=timestamp,
            scene=SCENE_NAME,
            event_type=EVENT_TYPE,
            risk_level=RISK_LEVEL,
            confidence=detection["score"],
            screenshot_path=screenshot_path,
            track_id=track_id,
            class_name=detection["class_name"]
        )
    except Exception as error:
        print(f"SQLite 事件记录写入失败：{error}")
        return None


def get_recent_events(limit=10):
    """查询最近的事件，并检查截图文件当前是否仍然存在。"""
    safe_limit = max(1, min(int(limit), 100))

    with database_lock:
        connection = sqlite3.connect(DATABASE_PATH)
        connection.row_factory = sqlite3.Row

        try:
            rows = connection.execute(
                """
                SELECT
                    id,
                    event_time,
                    scene,
                    event_type,
                    risk_level,
                    confidence,
                    screenshot_path,
                    track_id,
                    class_name
                FROM events
                ORDER BY id DESC
                LIMIT ?
                """,
                (safe_limit,)
            ).fetchall()
        finally:
            connection.close()

    events = []

    for row in rows:
        screenshot_path = row["screenshot_path"] or ""
        screenshot_exists = bool(
            screenshot_path
            and os.path.isfile(screenshot_path)
        )

        item = dict(row)
        item["screenshot_exists"] = screenshot_exists
        item["screenshot_url"] = (
            f"/evidence/{os.path.basename(screenshot_path)}"
            if screenshot_exists
            else None
        )
        events.append(item)

    return events

def validate_normalized_points(
    points,
    field_name,
    minimum_count=None,
    exact_count=None
):
    """校验网页传来的 0 到 1 比例坐标。"""
    if not isinstance(points, list):
        raise ValueError(f"{field_name} 必须是坐标列表")

    if exact_count is not None and len(points) != exact_count:
        raise ValueError(
            f"{field_name} 必须包含 {exact_count} 个点"
        )

    if minimum_count is not None and len(points) < minimum_count:
        raise ValueError(
            f"{field_name} 至少需要 {minimum_count} 个点"
        )

    checked_points = []

    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(
                f"{field_name} 第 {index + 1} 个点必须写成 [x, y]"
            )

        x = float(point[0])
        y = float(point[1])

        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(
                f"{field_name} 的坐标必须处于 0 到 1 之间"
            )

        checked_points.append([round(x, 6), round(y, 6)])

    return checked_points


def ensure_roi_config():
    """首次运行时创建同时包含 ROI 与入口线的规则配置。"""
    if os.path.exists(ROI_CONFIG_PATH):
        return

    initial_config = {
        "roi_points": validate_normalized_points(
            DEFAULT_ROI_POINTS,
            "roi_points",
            minimum_count=3
        ),
        "entry_line_points": validate_normalized_points(
            ENTRY_LINE_POINTS,
            "entry_line_points",
            exact_count=2
        )
    }

    with open(ROI_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(
            initial_config,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"已创建默认规则配置：{ROI_CONFIG_PATH}")


def load_roi_config_if_changed():
    """配置文件改变时，同时重载 ROI 和入口线。"""
    global roi_points_normalized
    global entry_line_points_normalized
    global roi_config_mtime

    try:
        current_mtime = os.path.getmtime(ROI_CONFIG_PATH)

        with roi_lock:
            if roi_config_mtime == current_mtime:
                return

            with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as file:
                config = json.load(file)

            checked_roi = validate_normalized_points(
                config.get("roi_points"),
                "roi_points",
                minimum_count=3
            )

            # 兼容原先只保存 ROI 的旧配置文件。
            checked_entry_line = validate_normalized_points(
                config.get("entry_line_points", ENTRY_LINE_POINTS),
                "entry_line_points",
                exact_count=2
            )

            line_dx = (
                checked_entry_line[1][0]
                - checked_entry_line[0][0]
            )
            line_dy = (
                checked_entry_line[1][1]
                - checked_entry_line[0][1]
            )
            if line_dx * line_dx + line_dy * line_dy < 1e-6:
                raise ValueError("入口线两个端点不能重合")

            roi_points_normalized = checked_roi
            entry_line_points_normalized = checked_entry_line
            roi_config_mtime = current_mtime

        print(
            f"规则配置已加载：ROI={checked_roi}，"
            f"入口线={checked_entry_line}"
        )

    except Exception as error:
        print(f"规则配置读取失败，继续使用上一次配置：{error}")


def get_rule_config():
    """返回网页编辑器需要的当前比例坐标。"""
    load_roi_config_if_changed()

    with roi_lock:
        return {
            "roi_points": [
                list(point) for point in roi_points_normalized
            ],
            "entry_line_points": [
                list(point)
                for point in entry_line_points_normalized
            ]
        }


def save_rule_config(roi_points, entry_line_points):
    """原子保存网页编辑后的 ROI 与入口线，并立即更新运行状态。"""
    global roi_points_normalized
    global entry_line_points_normalized
    global roi_config_mtime

    checked_roi = validate_normalized_points(
        roi_points,
        "roi_points",
        minimum_count=3
    )
    checked_entry_line = validate_normalized_points(
        entry_line_points,
        "entry_line_points",
        exact_count=2
    )

    line_dx = checked_entry_line[1][0] - checked_entry_line[0][0]
    line_dy = checked_entry_line[1][1] - checked_entry_line[0][1]
    if line_dx * line_dx + line_dy * line_dy < 1e-6:
        raise ValueError("入口线两个端点不能重合")

    config = {
        "roi_points": checked_roi,
        "entry_line_points": checked_entry_line
    }
    temporary_path = ROI_CONFIG_PATH + ".tmp"

    with roi_lock:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(
                config,
                file,
                ensure_ascii=False,
                indent=2
            )
            file.flush()
            os.fsync(file.fileno())

        os.replace(temporary_path, ROI_CONFIG_PATH)

        roi_points_normalized = checked_roi
        entry_line_points_normalized = checked_entry_line
        roi_config_mtime = os.path.getmtime(ROI_CONFIG_PATH)

    reset_tracking_state()
    print(
        f"网页规则已保存：ROI={checked_roi}，"
        f"入口线={checked_entry_line}"
    )

    return config


def get_roi_polygon(frame_shape):
    load_roi_config_if_changed()

    frame_height, frame_width = frame_shape[:2]

    with roi_lock:
        points = [list(point) for point in roi_points_normalized]

    polygon = np.array(
        [
            [
                int(round(x * (frame_width - 1))),
                int(round(y * (frame_height - 1)))
            ]
            for x, y in points
        ],
        dtype=np.int32
    )

    return polygon

def point_status(center_x, center_y, polygon):
    result = cv2.pointPolygonTest(
        polygon,
        (float(center_x), float(center_y)),
        False
    )

    return "inside" if result >= 0 else "outside"


def apply_roi_status(detections, frame_shape):
    polygon = get_roi_polygon(frame_shape)
    result = []

    for item in detections:
        updated = dict(item)

        box_height = updated["y2"] - updated["y1"]
        center_x = (updated["x1"] + updated["x2"]) // 2
        center_y = updated["y1"] + int(box_height * 0.85)
        status = point_status(center_x, center_y, polygon)

        # 第 8 步修正版：
        # 真正使用画面中的入口线判断“外侧”和“内侧”。
        line_start, line_end = get_entry_line(frame_shape)

        line_dx = line_end[0] - line_start[0]
        line_dy = line_end[1] - line_start[1]

        point_side_value = (
            line_dx * (center_y - line_start[1])
            - line_dy * (center_x - line_start[0])
        )

        # 用 ROI 中心点所在的一侧定义为“楼内侧”，
        # 因此入口线两端顺序调整后也不会把方向弄反。
        roi_center_x = float(np.mean(polygon[:, 0]))
        roi_center_y = float(np.mean(polygon[:, 1]))

        roi_side_value = (
            line_dx * (roi_center_y - line_start[1])
            - line_dy * (roi_center_x - line_start[0])
        )

        same_side_as_roi = (
            point_side_value * roi_side_value >= 0
        )

        line_status = (
            "inside"
            if same_side_as_roi
            else "outside"
        )

        updated["center_x"] = center_x
        updated["center_y"] = center_y
        updated["roi_status"] = status
        updated["line_status"] = line_status

        result.append(updated)

    return result


def get_entry_line(frame_shape):
    """根据网页可编辑的比例坐标生成入口线。"""
    load_roi_config_if_changed()
    frame_height, frame_width = frame_shape[:2]

    with roi_lock:
        points = [
            list(point) for point in entry_line_points_normalized
        ]

    line_points = []
    for x, y in points:
        line_points.append((
            int(round(x * (frame_width - 1))),
            int(round(y * (frame_height - 1)))
        ))

    return line_points[0], line_points[1]


def point_to_segment_distance(point, line_start, line_end):
    """计算中心点到入口线段的距离，用于目标匹配和显示。"""
    point_array = np.array(point, dtype=np.float32)
    start_array = np.array(line_start, dtype=np.float32)
    end_array = np.array(line_end, dtype=np.float32)

    segment = end_array - start_array
    segment_length_squared = float(np.dot(segment, segment))

    if segment_length_squared <= 1e-6:
        return float(np.linalg.norm(point_array - start_array))

    ratio = float(
        np.dot(point_array - start_array, segment)
        / segment_length_squared
    )
    ratio = max(0.0, min(1.0, ratio))
    nearest = start_array + ratio * segment

    return float(np.linalg.norm(point_array - nearest))


def _match_detections_to_tracks_unlocked(detections, evidence_frame=None):
    """
    第 8 步稳定版报警逻辑：

    1. 入口线仍用于判断 ENTER 和 EXIT 方向。
    2. 只要电动车连续 CONFIRM_HIT_FRAMES 次位于 ROI 内，
       就确认一次违规事件并触发 LED。
    3. 目标持续留在 ROI 内时只报警一次。
    4. 目标离开 ROI 后，下一次进入可以重新报警。
    5. 全系统使用 EVENT_COOLDOWN_SECONDS 冷却时间。
    """
    global tracks
    global next_track_id
    global event_count
    global latest_event_text
    global latest_event_time
    global last_alarm_time

    now = time.time()
    unmatched_track_ids = set(tracks.keys())
    used_track_ids = set()
    updated_detections = []

    sorted_detections = sorted(
        detections,
        key=lambda item: item["score"],
        reverse=True
    )

    for detection in sorted_detections:
        center = (
            detection["center_x"],
            detection["center_y"]
        )

        best_track_id = None
        best_distance = TRACK_MAX_DISTANCE

        for track_id in list(unmatched_track_ids):
            track = tracks[track_id]

            if track["class_name"] != detection["class_name"]:
                continue

            last_center = track["center"]
            distance = float(
                np.hypot(
                    center[0] - last_center[0],
                    center[1] - last_center[1]
                )
            )

            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = next_track_id
            next_track_id += 1

            tracks[best_track_id] = {
                "class_name": detection["class_name"],
                "center": center,
                "previous_center": None,

                # 入口线两侧状态，只负责方向判断。
                "line_status": detection["line_status"],
                "previous_line_status": None,

                # ROI 状态负责稳定违规确认。
                "roi_status": detection["roi_status"],
                "previous_roi_status": None,
                "roi_confirm_count": 0,
                "violation_alarm_sent": False,
                "cooldown_notice_sent": False,

                "missed": 0,
                "history": [center],
                "enter_count": 0
            }
        else:
            unmatched_track_ids.discard(best_track_id)

        track = tracks[best_track_id]

        previous_center = track["center"]
        previous_line_status = track["line_status"]
        previous_roi_status = track["roi_status"]

        current_line_status = detection["line_status"]
        current_roi_status = detection["roi_status"]

        track["previous_center"] = previous_center
        track["center"] = center

        track["previous_line_status"] = previous_line_status
        track["line_status"] = current_line_status

        track["previous_roi_status"] = previous_roi_status
        track["roi_status"] = current_roi_status

        track["missed"] = 0
        track["history"].append(center)
        track["history"] = track["history"][-POSITION_HISTORY_LENGTH:]

        direction = "none"
        enter_event = False
        event_suppressed = False

        # 入口线方向判断继续保留，方便现场展示。
        if (
            previous_line_status == "outside"
            and current_line_status == "inside"
        ):
            direction = "enter"
            print(
                f"ENTER：目标 ID {best_track_id}，"
                f"跨过入口线进入楼内侧"
            )

        elif (
            previous_line_status == "inside"
            and current_line_status == "outside"
        ):
            direction = "exit"
            print(
                f"EXIT：目标 ID {best_track_id}，"
                f"跨过入口线离开楼内侧"
            )

        # 稳定报警判断使用 ROI。
        if current_roi_status == "inside":
            track["roi_confirm_count"] += 1

            if (
                not track["violation_alarm_sent"]
                and track["roi_confirm_count"] <= CONFIRM_HIT_FRAMES
            ):
                print(
                    f"违规确认：目标 ID {best_track_id}，"
                    f"ROI 内连续命中 "
                    f"{track['roi_confirm_count']}/"
                    f"{CONFIRM_HIT_FRAMES}"
                )

            if (
                not track["violation_alarm_sent"]
                and track["roi_confirm_count"] >= CONFIRM_HIT_FRAMES
            ):
                cooldown_elapsed = now - last_alarm_time

                if cooldown_elapsed >= EVENT_COOLDOWN_SECONDS:
                    enter_event = True
                    track["violation_alarm_sent"] = True
                    track["cooldown_notice_sent"] = False
                    track["enter_count"] += 1

                    event_count += 1
                    latest_event_time = now
                    last_alarm_time = now
                    latest_event_text = (
                        f"ALARM  ID {best_track_id}  "
                        f"{detection['class_name']}"
                    )

                    append_event_log(
                        timestamp=now,
                        event_type=EVENT_TYPE,
                        track_id=best_track_id,
                        class_name=detection["class_name"],
                        confidence=detection["score"]
                    )

                    event_id = save_event_record(
                        frame=evidence_frame,
                        detection=detection,
                        track_id=best_track_id,
                        timestamp=now
                    )

                    led_started = alarm.trigger(
                        reason=(
                            f"目标 ID {best_track_id}，"
                            f"类别 {detection['class_name']}，"
                            f"ROI 内连续确认 "
                            f"{track['roi_confirm_count']} 次"
                        )
                    )

                    print("========== ALARM START ==========")
                    print(
                        f"报警开始时间："
                        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}"
                    )
                    print(f"事件类型：{EVENT_TYPE}")
                    print(
                        f"目标 ID：{best_track_id}，"
                        f"类别：{detection['class_name']}，"
                        f"ROI 连续命中："
                        f"{track['roi_confirm_count']} 次，"
                        f"累计报警：{event_count}"
                    )
                    print(f"LED触发结果：{led_started}")
                    print(f"CSV 日志：{EVENT_LOG_PATH}")
                    print(f"SQLite 事件编号：{event_id}")
                    print(f"SQLite 数据库：{DATABASE_PATH}")
                    print("=================================")

                else:
                    remaining = (
                        EVENT_COOLDOWN_SECONDS - cooldown_elapsed
                    )
                    event_suppressed = True

                    if not track["cooldown_notice_sent"]:
                        print(
                            f"冷却期内暂缓报警："
                            f"目标 ID {best_track_id}，"
                            f"剩余 {remaining:.1f} 秒"
                        )
                        track["cooldown_notice_sent"] = True

        else:
            # 离开 ROI 后，允许下一次进入重新完成三次确认。
            if previous_roi_status == "inside":
                print(
                    f"目标 ID {best_track_id} 已离开 ROI，"
                    f"报警状态已复位"
                )

            track["roi_confirm_count"] = 0
            track["violation_alarm_sent"] = False
            track["cooldown_notice_sent"] = False

        pending_enter = (
            current_roi_status == "inside"
            and not track["violation_alarm_sent"]
            and track["roi_confirm_count"] < CONFIRM_HIT_FRAMES
        )

        updated = dict(detection)
        updated["track_id"] = best_track_id
        updated["previous_center"] = previous_center
        updated["previous_roi_status"] = previous_roi_status
        updated["direction"] = direction
        updated["enter_event"] = enter_event
        updated["event_suppressed"] = event_suppressed
        updated["confirm_count"] = track["roi_confirm_count"]
        updated["pending_enter"] = pending_enter
        updated["track_history"] = list(track["history"])

        updated_detections.append(updated)
        used_track_ids.add(best_track_id)

    # 漏检会打断尚未完成的连续确认。
    for track_id in list(tracks.keys()):
        if track_id in used_track_ids:
            continue

        track = tracks[track_id]
        track["missed"] += 1

        if (
            track["missed"] > CONFIRM_MISS_TOLERANCE
            and track["roi_confirm_count"] > 0
            and not track["violation_alarm_sent"]
        ):
            track["roi_confirm_count"] = 0
            track["cooldown_notice_sent"] = False
            print(
                f"违规确认中断：目标 ID {track_id} "
                f"连续漏检超过 {CONFIRM_MISS_TOLERANCE} 次"
            )

        if track["missed"] > TRACK_MAX_MISSED:
            del tracks[track_id]

    updated_detections.sort(
        key=lambda item: (item["center_x"], item["center_y"])
    )

    return updated_detections


def match_detections_to_tracks(detections, evidence_frame=None):
    """串行更新目标轨迹，避免网页保存规则时与检测线程冲突。"""
    with tracking_lock:
        return _match_detections_to_tracks_unlocked(
            detections,
            evidence_frame=evidence_frame
        )


def letterbox(image, size=640):
    image_height, image_width = image.shape[:2]
    scale = min(size / image_width, size / image_height)

    new_width = int(round(image_width * scale))
    new_height = int(round(image_height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_width = size - new_width
    pad_height = size - new_height

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    return padded, scale, pad_left, pad_top


def xywh_to_xyxy(boxes):
    result = np.empty_like(boxes)

    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return result


def postprocess(output, scale, pad_left, pad_top, original_shape):
    prediction = output[0]

    if prediction.shape[0] == 84:
        prediction = prediction.transpose(1, 0)

    if prediction.ndim != 2 or prediction.shape[1] < 5:
        raise RuntimeError(f"模型输出形状异常：{prediction.shape}")

    boxes = prediction[:, :4]
    class_scores = prediction[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    confidence_mask = confidences >= CONF_THRESHOLD
    boxes = boxes[confidence_mask]
    class_ids = class_ids[confidence_mask]
    confidences = confidences[confidence_mask]

    if len(boxes) == 0:
        return []

    valid_class_mask = class_ids < len(CLASS_NAMES)
    boxes = boxes[valid_class_mask]
    class_ids = class_ids[valid_class_mask]
    confidences = confidences[valid_class_mask]

    target_mask = np.array([
        CLASS_NAMES[int(class_id)] in TARGET_CLASSES
        for class_id in class_ids
    ])

    boxes = boxes[target_mask]
    class_ids = class_ids[target_mask]
    confidences = confidences[target_mask]

    if len(boxes) == 0:
        return []

    boxes = xywh_to_xyxy(boxes)

    boxes[:, [0, 2]] -= pad_left
    boxes[:, [1, 3]] -= pad_top
    boxes /= scale

    image_height, image_width = original_shape[:2]

    boxes[:, 0] = np.clip(boxes[:, 0], 0, image_width - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, image_height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, image_width - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, image_height - 1)

    nms_boxes = []

    for box in boxes:
        x1, y1, x2, y2 = box
        nms_boxes.append([
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1)
        ])

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        confidences.tolist(),
        CONF_THRESHOLD,
        NMS_THRESHOLD
    )

    if len(indices) == 0:
        return []

    detections = []

    for index in np.array(indices).reshape(-1):
        x1, y1, x2, y2 = boxes[index].astype(int)
        class_id = int(class_ids[index])

        detections.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "score": float(confidences[index]),
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        })

    return detections


def open_input_source():
    if SOURCE_MODE == "video":
        capture = cv2.VideoCapture(VIDEO_PATH)

        if not capture.isOpened():
            raise RuntimeError(f"无法打开测试视频：{VIDEO_PATH}")

        fps = capture.get(cv2.CAP_PROP_FPS)

        if fps <= 1 or fps > 120:
            fps = 25.0

        print(f"测试视频打开成功：{VIDEO_PATH}")
        print(
            f"视频分辨率："
            f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
            f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        )
        print(f"视频帧率：{fps:.1f}")

        return capture, fps

    capture = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    capture.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(f"无法打开摄像头：{CAMERA_DEVICE}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 1 or fps > 120:
        fps = CAMERA_FPS

    print("摄像头打开成功")
    print(f"设备：{CAMERA_DEVICE}")
    print(
        f"实际分辨率："
        f"{int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))} x "
        f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    )
    print(f"摄像头帧率：{fps:.1f}")

    return capture, fps


ensure_roi_config()
ensure_event_log()
ensure_event_database()

print("正在加载 YOLOv8n RKNN 模型...")

rknn = RKNNLite()

ret = rknn.load_rknn(MODEL_PATH)

if ret != 0:
    raise RuntimeError(f"RKNN 模型加载失败，错误码：{ret}")

print("RKNN 模型加载成功")

ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)

if ret != 0:
    raise RuntimeError(f"RK3588 NPU 初始化失败，错误码：{ret}")

print("RK3588 NPU 初始化成功")

cap, source_fps = open_input_source()

if ALARM_MODE == "led":
    alarm = LEDAlarm(
        chip=LED_GPIO_CHIP,
        line=LED_GPIO_LINE,
        alarm_duration=LED_ALARM_DURATION_SECONDS
    )
else:
    alarm = FakeAlarm()


def detect_objects(frame):
    preprocess_start = time.perf_counter()

    input_image, scale, pad_left, pad_top = letterbox(
        frame,
        INPUT_SIZE
    )

    input_rgb = cv2.cvtColor(
        input_image,
        cv2.COLOR_BGR2RGB
    )

    input_data = np.expand_dims(input_rgb, axis=0)

    preprocess_ms = (
        time.perf_counter() - preprocess_start
    ) * 1000

    inference_start = time.perf_counter()

    outputs = rknn.inference(inputs=[input_data])

    inference_ms = (
        time.perf_counter() - inference_start
    ) * 1000

    if outputs is None:
        raise RuntimeError("RKNN 推理失败，返回结果为空")

    postprocess_start = time.perf_counter()

    detections = postprocess(
        outputs[0],
        scale,
        pad_left,
        pad_top,
        frame.shape
    )

    detections = apply_roi_status(
        detections,
        frame.shape
    )

    detections = match_detections_to_tracks(
        detections,
        evidence_frame=frame
    )

    postprocess_ms = (
        time.perf_counter() - postprocess_start
    ) * 1000

    return (
        detections,
        preprocess_ms,
        inference_ms,
        postprocess_ms
    )


def draw_result(
    frame,
    detections,
    preprocess_ms,
    inference_ms,
    postprocess_ms,
    display_fps
):
    result = frame.copy()
    polygon = get_roi_polygon(result.shape)

    overlay = result.copy()

    cv2.fillPoly(
        overlay,
        [polygon],
        (0, 165, 255)
    )

    result = cv2.addWeighted(
        overlay,
        0.18,
        result,
        0.82,
        0
    )

    cv2.polylines(
        result,
        [polygon],
        True,
        (0, 165, 255),
        3
    )

    first_point = tuple(polygon[0])

    cv2.putText(
        result,
        "ROI",
        (first_point[0], max(28, first_point[1] - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 165, 255),
        2
    )

    line_start, line_end = get_entry_line(result.shape)

    cv2.line(
        result,
        line_start,
        line_end,
        (255, 255, 0),
        4
    )

    line_label_x = min(line_start[0], line_end[0])
    line_label_y = max(28, min(line_start[1], line_end[1]) - 12)

    cv2.putText(
        result,
        "ENTRY LINE",
        (line_label_x, line_label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2
    )

    fps_text = f"FPS: {display_fps:.1f}"
    fps_text_size, _ = cv2.getTextSize(
        fps_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        2
    )
    fps_x = max(20, result.shape[1] - fps_text_size[0] - 20)
    cv2.putText(
        result,
        fps_text,
        (fps_x, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    inside_count = 0
    max_confirm_count = 0

    for item in detections:
        x1 = item["x1"]
        y1 = item["y1"]
        x2 = item["x2"]
        y2 = item["y2"]

        center_x = item["center_x"]
        center_y = item["center_y"]
        status = item["roi_status"]
        track_id = item.get("track_id", -1)
        direction = item.get("direction", "none")
        enter_event = item.get("enter_event", False)
        event_suppressed = item.get("event_suppressed", False)
        confirm_count = item.get("confirm_count", 0)
        max_confirm_count = max(
            max_confirm_count,
            min(confirm_count, CONFIRM_HIT_FRAMES)
        )
        pending_enter = item.get("pending_enter", False)
        history = item.get("track_history", [])

        if status == "inside":
            color = (0, 0, 255)
            inside_count += 1
        else:
            color = (0, 255, 0)

        if enter_event:
            color = (255, 0, 255)

        line_status = item.get("line_status", "unknown")

        label = (
            f"ID {track_id} "
            f"{item['class_name']} "
            f"{item['score']:.2f} "
            f"ROI:{status} "
            f"LINE:{line_status}"
        )

        if direction in {"enter", "exit"}:
            label += f" {direction.upper()}"


        if event_suppressed:
            label += " COOLDOWN"

        if len(history) >= 2:
            for history_index in range(1, len(history)):
                cv2.line(
                    result,
                    tuple(history[history_index - 1]),
                    tuple(history[history_index]),
                    color,
                    2
                )

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        cv2.circle(
            result,
            (center_x, center_y),
            6,
            color,
            -1
        )

        cv2.putText(
            result,
            label,
            (x1, max(22, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        result,
        f"NPU: {inference_ms:.0f} ms",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Objects: {len(detections)}",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Inside: {inside_count}",
        (20, 101),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Confirm: {max_confirm_count}/{CONFIRM_HIT_FRAMES}",
        (20, 134),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    total_ms = preprocess_ms + inference_ms + postprocess_ms

    cv2.putText(
        result,
        f"Total: {total_ms:.0f} ms",
        (20, 167),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cooldown_remaining = max(
        0.0,
        EVENT_COOLDOWN_SECONDS - (time.time() - last_alarm_time)
    )

    cv2.putText(
        result,
        f"Alarms: {event_count}  Cooldown: {cooldown_remaining:.1f}s",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    event_age = time.time() - latest_event_time

    if latest_event_time > 0 and event_age <= 3.0:
        cv2.rectangle(
            result,
            (15, 218),
            (min(result.shape[1] - 15, 520), 258),
            (0, 0, 255),
            -1
        )
        cv2.putText(
            result,
            latest_event_text,
            (25, 246),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

    return result


def camera_loop():
    global latest_raw_frame
    global latest_display_frame
    global latest_display_fps

    frame_delay = 1.0 / max(source_fps, 1.0)
    fps_window_start = time.perf_counter()
    fps_frame_count = 0

    while running:
        frame_start = time.perf_counter()
        success, frame = cap.read()

        if not success or frame is None:
            if SOURCE_MODE == "video" and LOOP_VIDEO:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                reset_tracking_state()
                print("测试视频播放完毕，正在从头循环")
                time.sleep(0.2)
                continue

            print("读取输入画面失败")
            time.sleep(0.2)
            continue

        fps_frame_count += 1
        fps_elapsed = time.perf_counter() - fps_window_start
        if fps_elapsed >= 1.0:
            latest_display_fps = fps_frame_count / fps_elapsed
            fps_frame_count = 0
            fps_window_start = time.perf_counter()

        with frame_lock:
            latest_raw_frame = frame.copy()
            detections = list(latest_detections)
            preprocess_ms = latest_preprocess_ms
            inference_ms = latest_inference_ms
            postprocess_ms = latest_postprocess_ms
            display_fps = latest_display_fps

        display_frame = draw_result(
            frame,
            detections,
            preprocess_ms,
            inference_ms,
            postprocess_ms,
            display_fps
        )

        with frame_lock:
            latest_display_frame = display_frame

        if SOURCE_MODE == "video":
            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, frame_delay - elapsed))
        else:
            # 摄像头 read() 本身会按设备帧率阻塞，只做极短让步。
            time.sleep(0.001)


def detection_loop():
    global latest_detections
    global latest_preprocess_ms
    global latest_inference_ms
    global latest_postprocess_ms

    last_performance_log_time = 0.0
    empty_detection_cycles = 0

    while running:
        cycle_start = time.perf_counter()

        with frame_lock:
            frame = (
                None
                if latest_raw_frame is None
                else latest_raw_frame.copy()
            )

        if frame is not None:
            try:
                (
                    detections,
                    preprocess_ms,
                    inference_ms,
                    postprocess_ms
                ) = detect_objects(frame)

                with frame_lock:
                    if detections:
                        latest_detections = detections
                        empty_detection_cycles = 0
                    else:
                        empty_detection_cycles += 1

                        if empty_detection_cycles > DETECTION_HOLD_CYCLES:
                            latest_detections = []

                    latest_preprocess_ms = preprocess_ms
                    latest_inference_ms = inference_ms
                    latest_postprocess_ms = postprocess_ms

                total_ms = preprocess_ms + inference_ms + postprocess_ms
                log_now = time.perf_counter()

                if (
                    log_now - last_performance_log_time
                    >= PERFORMANCE_LOG_INTERVAL
                ):
                    print(
                        f"前处理：{preprocess_ms:.1f} ms，"
                        f"NPU推理：{inference_ms:.1f} ms，"
                        f"后处理：{postprocess_ms:.1f} ms，"
                        f"总耗时：{total_ms:.1f} ms，"
                        f"检测数量：{len(detections)}，"
                        f"画面FPS：{latest_display_fps:.1f}"
                    )

                    for item in detections:
                        print(
                            f"类别：{item['class_name']}，"
                            f"置信度：{item['score']:.2f}，"
                            f"中心点："
                            f"({item['center_x']}, {item['center_y']})，"
                            f"ROI状态：{item['roi_status']}，"
                            f"入口线侧：{item.get('line_status')}，"
                            f"目标ID：{item.get('track_id')}，"
                            f"方向：{item.get('direction')}"
                        )

                    last_performance_log_time = log_now

            except Exception as error:
                print(f"RKNN 模型推理失败：{error}")

        elapsed = time.perf_counter() - cycle_start
        time.sleep(max(0.001, DETECT_PERIOD - elapsed))


def generate_frames():
    stream_delay = 1.0 / max(WEB_STREAM_FPS, 1.0)

    while running:
        stream_start = time.perf_counter()

        with frame_lock:
            frame = (
                None
                if latest_display_frame is None
                else latest_display_frame.copy()
            )

        if frame is None:
            time.sleep(0.03)
            continue

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        )

        if not success:
            time.sleep(0.03)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

        elapsed = time.perf_counter() - stream_start
        time.sleep(max(0.001, stream_delay - elapsed))


PAGE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RK3588 电动车违规事件管理</title>
    <style>
        * {
            box-sizing: border-box;
        }
        body {
            margin: 0;
            padding: 6px;
            text-align: center;
            font-family: Arial, sans-serif;
            background-color: #202124;
            color: white;
        }
        h1 {
            margin: 4px 0 2px;
            font-size: 22px;
        }
        p {
            color: #cccccc;
            margin: 2px 0;
            font-size: 14px;
        }
        .editor-shell {
            position: relative;
            width: min(1280px, 99vw);
            margin: 4px auto 0;
            border: 2px solid #777777;
            border-radius: 8px;
            overflow: hidden;
            line-height: 0;
            background: #111111;
        }
        .video {
            display: block;
            width: 100%;
            height: auto;
        }
        #rule-canvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            touch-action: none;
            cursor: default;
        }
        .controls {
            margin: 8px auto 2px;
        }
        button {
            margin: 4px 6px;
            padding: 10px 20px;
            border: 1px solid #777777;
            border-radius: 6px;
            font-size: 16px;
            color: white;
            background: #3a3d42;
            cursor: pointer;
        }
        button:hover {
            background: #4a4d52;
        }
        button.active {
            border-color: #ffb74d;
            background: #6d4c22;
        }
        #message {
            min-height: 26px;
            margin-top: 8px;
            color: #d7e3fc;
        }
        .events {
            max-width: 1200px;
            margin: 10px auto 0;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #2b2d31;
        }
        th, td {
            border: 1px solid #555555;
            padding: 9px;
            text-align: center;
            white-space: nowrap;
        }
        th {
            background: #3a3d42;
        }
        .thumb {
            width: 150px;
            max-height: 100px;
            object-fit: cover;
            border-radius: 4px;
        }
        .missing {
            color: #ffb74d;
        }
    </style>
</head>
<body>
    <h1>RK3588 电动车违规事件管理</h1>
    <p>拖动顶点调整检测区域，保存后主程序立即使用新坐标</p>

    <div class="editor-shell" id="editor-shell">
        <img id="video" class="video" src="/video_feed" alt="ROI 检测画面">
        <canvas id="rule-canvas"></canvas>
    </div>

    <div class="controls">
        <button id="edit-roi" onclick="setEditMode('roi')">编辑电子围栏</button>
        <button id="edit-line" onclick="setEditMode('line')">编辑入口线</button>
        <button onclick="saveRules()">保存配置</button>
    </div>
    <div id="message">正在读取当前配置……</div>

    <div class="events">
        <h2>最近 10 条违规事件</h2>
        <table>
            <thead>
                <tr>
                    <th>编号</th>
                    <th>时间</th>
                    <th>场景</th>
                    <th>事件类型</th>
                    <th>风险等级</th>
                    <th>类别</th>
                    <th>置信度</th>
                    <th>截图</th>
                </tr>
            </thead>
            <tbody id="event-body">
                <tr><td colspan="8">正在读取……</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("rule-canvas");
        const context = canvas.getContext("2d");
        const message = document.getElementById("message");

        let roiPoints = [];
        let entryLinePoints = [];
        let editMode = null;
        let draggingIndex = -1;

        function clamp(value, minimum, maximum) {
            return Math.max(minimum, Math.min(maximum, value));
        }

        function resizeCanvas() {
            const width = Math.max(1, Math.round(video.clientWidth));
            const height = Math.max(1, Math.round(video.clientHeight));

            if (canvas.width !== width || canvas.height !== height) {
                canvas.width = width;
                canvas.height = height;
            }

            drawEditor();
        }

        function toCanvasPoint(point) {
            return {
                x: point[0] * canvas.width,
                y: point[1] * canvas.height
            };
        }

        function drawHandle(point, fillColor) {
            const position = toCanvasPoint(point);
            context.beginPath();
            context.arc(position.x, position.y, 9, 0, Math.PI * 2);
            context.fillStyle = fillColor;
            context.fill();
            context.lineWidth = 3;
            context.strokeStyle = "#ffffff";
            context.stroke();
        }

        function drawEditor() {
            context.clearRect(0, 0, canvas.width, canvas.height);

            if (editMode === "roi" && roiPoints.length >= 3) {
                context.beginPath();
                const first = toCanvasPoint(roiPoints[0]);
                context.moveTo(first.x, first.y);

                roiPoints.slice(1).forEach(point => {
                    const position = toCanvasPoint(point);
                    context.lineTo(position.x, position.y);
                });

                context.closePath();
                context.fillStyle = "rgba(255, 165, 0, 0.20)";
                context.fill();
                context.lineWidth = 4;
                context.strokeStyle = "#ff9800";
                context.stroke();

                roiPoints.forEach(point => {
                    drawHandle(point, "#ff9800");
                });
            }

            if (editMode === "line" && entryLinePoints.length === 2) {
                const start = toCanvasPoint(entryLinePoints[0]);
                const end = toCanvasPoint(entryLinePoints[1]);

                context.beginPath();
                context.moveTo(start.x, start.y);
                context.lineTo(end.x, end.y);
                context.lineWidth = 5;
                context.strokeStyle = "#00e5ff";
                context.stroke();

                entryLinePoints.forEach(point => {
                    drawHandle(point, "#00b8d4");
                });
            }
        }

        function updateButtonState() {
            document.getElementById("edit-roi")
                .classList.toggle("active", editMode === "roi");
            document.getElementById("edit-line")
                .classList.toggle("active", editMode === "line");
        }

        function setEditMode(mode) {
            editMode = mode;
            draggingIndex = -1;
            canvas.style.pointerEvents = "auto";
            canvas.style.cursor = "grab";
            updateButtonState();
            resizeCanvas();

            if (mode === "roi") {
                message.textContent = "拖动橙色圆点，调整电子围栏范围";
            } else {
                message.textContent = "拖动蓝色圆点，调整入口线位置和角度";
            }
        }

        function pointerPosition(event) {
            const rectangle = canvas.getBoundingClientRect();
            return {
                x: (event.clientX - rectangle.left)
                    * canvas.width / rectangle.width,
                y: (event.clientY - rectangle.top)
                    * canvas.height / rectangle.height
            };
        }

        function editablePoints() {
            return editMode === "roi" ? roiPoints : entryLinePoints;
        }

        canvas.addEventListener("pointerdown", event => {
            if (!editMode) {
                return;
            }

            const position = pointerPosition(event);
            const points = editablePoints();
            let closestDistance = 22;
            draggingIndex = -1;

            points.forEach((point, index) => {
                const handle = toCanvasPoint(point);
                const distance = Math.hypot(
                    position.x - handle.x,
                    position.y - handle.y
                );

                if (distance < closestDistance) {
                    closestDistance = distance;
                    draggingIndex = index;
                }
            });

            if (draggingIndex >= 0) {
                canvas.setPointerCapture(event.pointerId);
                canvas.style.cursor = "grabbing";
                event.preventDefault();
            }
        });

        canvas.addEventListener("pointermove", event => {
            if (draggingIndex < 0 || !editMode) {
                return;
            }

            const position = pointerPosition(event);
            const points = editablePoints();
            points[draggingIndex] = [
                clamp(position.x / canvas.width, 0, 1),
                clamp(position.y / canvas.height, 0, 1)
            ];
            drawEditor();
            event.preventDefault();
        });

        function finishDragging(event) {
            if (draggingIndex >= 0) {
                draggingIndex = -1;
                canvas.style.cursor = "grab";

                if (canvas.hasPointerCapture(event.pointerId)) {
                    canvas.releasePointerCapture(event.pointerId);
                }
            }
        }

        canvas.addEventListener("pointerup", finishDragging);
        canvas.addEventListener("pointercancel", finishDragging);

        async function loadRules() {
            try {
                const response = await fetch("/api/rules");
                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(result.message || "读取配置失败");
                }

                roiPoints = result.roi_points.map(point => [
                    Number(point[0]),
                    Number(point[1])
                ]);
                entryLinePoints = result.entry_line_points.map(point => [
                    Number(point[0]),
                    Number(point[1])
                ]);
                message.textContent = "配置读取成功，请选择需要编辑的项目";
                resizeCanvas();
            } catch (error) {
                message.textContent = "配置读取失败：" + error;
            }
        }

        async function saveRules() {
            if (roiPoints.length < 3 || entryLinePoints.length !== 2) {
                message.textContent = "当前坐标不完整，无法保存";
                return;
            }

            message.textContent = "正在保存配置……";

            try {
                const response = await fetch("/api/rules", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        roi_points: roiPoints,
                        entry_line_points: entryLinePoints
                    })
                });
                const result = await response.json();

                if (!response.ok || !result.success) {
                    throw new Error(result.message || "保存失败");
                }

                roiPoints = result.roi_points;
                entryLinePoints = result.entry_line_points;
                editMode = null;
                draggingIndex = -1;
                canvas.style.pointerEvents = "none";
                canvas.style.cursor = "default";
                updateButtonState();
                drawEditor();
                message.textContent = "配置保存成功，检测程序已立即使用新坐标";
            } catch (error) {
                message.textContent = "保存失败：" + error;
            }
        }

        function escapeHtml(value) {
            return String(value ?? "")
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#039;");
        }

        async function loadEvents() {
            const body = document.getElementById("event-body");

            try {
                const response = await fetch("/api/events?limit=10");
                const result = await response.json();

                if (!result.success) {
                    throw new Error(result.message || "查询失败");
                }

                if (result.events.length === 0) {
                    body.innerHTML =
                        '<tr><td colspan="8">目前还没有报警记录</td></tr>';
                    return;
                }

                body.innerHTML = result.events.map(event => {
                    let screenshotCell =
                        '<span class="missing">截图已删除或保存失败</span>';

                    if (event.screenshot_exists) {
                        const safeUrl = escapeHtml(event.screenshot_url);
                        screenshotCell =
                            `<a href="${safeUrl}" target="_blank">` +
                            `<img class="thumb" src="${safeUrl}" ` +
                            `alt="事件截图">` +
                            `</a>`;
                    }

                    return `
                        <tr>
                            <td>${escapeHtml(event.id)}</td>
                            <td>${escapeHtml(event.event_time)}</td>
                            <td>${escapeHtml(event.scene)}</td>
                            <td>${escapeHtml(event.event_type)}</td>
                            <td>${escapeHtml(event.risk_level)}</td>
                            <td>${escapeHtml(event.class_name)}</td>
                            <td>${Number(event.confidence).toFixed(3)}</td>
                            <td>${screenshotCell}</td>
                        </tr>
                    `;
                }).join("");

                document.querySelectorAll(".thumb").forEach(image => {
                    image.addEventListener("error", () => {
                        const missing = document.createElement("span");
                        missing.className = "missing";
                        missing.textContent = "截图文件已不存在";
                        image.parentElement.replaceWith(missing);
                    });
                });
            } catch (error) {
                body.innerHTML =
                    `<tr><td colspan="8">读取失败：` +
                    `${escapeHtml(error)}</td></tr>`;
            }
        }

        video.addEventListener("load", resizeCanvas);
        window.addEventListener("resize", resizeCanvas);

        if (window.ResizeObserver) {
            const resizeObserver = new ResizeObserver(resizeCanvas);
            resizeObserver.observe(video);
        }

        loadRules();
        loadEvents();
        setInterval(loadEvents, 5000);
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


@app.route("/api/rules", methods=["GET", "POST"])
def api_rules():
    if request.method == "GET":
        try:
            config = get_rule_config()
            return jsonify(success=True, **config)
        except Exception as error:
            print(f"读取网页规则失败：{error}")
            return jsonify(
                success=False,
                message=str(error)
            ), 500

    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError("请求内容必须是 JSON 对象")

        config = save_rule_config(
            roi_points=data.get("roi_points"),
            entry_line_points=data.get("entry_line_points")
        )

        return jsonify(success=True, **config)

    except (TypeError, ValueError) as error:
        print(f"网页规则参数无效：{error}")
        return jsonify(
            success=False,
            message=str(error)
        ), 400

    except Exception as error:
        print(f"网页规则保存失败：{error}")
        return jsonify(
            success=False,
            message=str(error)
        ), 500


@app.route("/api/events")
def api_events():
    try:
        limit = request.args.get("limit", default=10, type=int)
        events = get_recent_events(limit=limit)
        return jsonify(success=True, events=events)
    except Exception as error:
        print(f"查询 SQLite 事件失败：{error}")
        return jsonify(
            success=False,
            message=str(error),
            events=[]
        ), 500


@app.route("/evidence/<path:filename>")
def evidence_file(filename):
    filepath = os.path.join(EVIDENCE_DIR, filename)

    if not os.path.isfile(filepath):
        return jsonify(
            success=False,
            message="截图文件不存在，事件日志仍然保留"
        ), 404

    return send_from_directory(
        EVIDENCE_DIR,
        filename,
        as_attachment=False
    )


@app.route("/snapshot", methods=["POST"])
def snapshot():
    with frame_lock:
        frame = (
            None
            if latest_display_frame is None
            else latest_display_frame.copy()
        )

    if frame is None:
        return jsonify(
            success=False,
            message="当前没有可保存画面"
        ), 503

    filename = time.strftime(
        "direction_snapshot_%Y%m%d_%H%M%S.jpg"
    )

    filepath = os.path.join(SAVE_DIR, filename)

    if not cv2.imwrite(filepath, frame):
        return jsonify(
            success=False,
            message="图片保存失败"
        ), 500

    print(f"图片已保存：{filepath}")

    return jsonify(
        success=True,
        filename=filename
    )


camera_thread = threading.Thread(
    target=camera_loop,
    daemon=True
)

detection_thread = threading.Thread(
    target=detection_loop,
    daemon=True
)

camera_thread.start()
detection_thread.start()


if __name__ == "__main__":
    print("========== 主程序启动成功 ==========")
    print(f"主配置文件：{MAIN_CONFIG_PATH}")
    print(f"输入模式：{SOURCE_MODE}")
    if SOURCE_MODE == "camera":
        print(
            f"摄像头：{CAMERA_DEVICE}，"
            f"请求分辨率：{CAMERA_WIDTH}x{CAMERA_HEIGHT}，"
            f"请求帧率：{CAMERA_FPS:.1f}"
        )
    else:
        print(f"测试视频：{VIDEO_PATH}，循环播放：{LOOP_VIDEO}")
    print(f"RKNN 模型：{MODEL_PATH}")
    print(f"浏览器访问：http://开发板IP:{WEB_PORT}")
    print(f"性能档位：{PERFORMANCE_MODE}")
    print(
        f"网页推流：{WEB_STREAM_FPS:.1f} FPS，"
        f"JPEG画质：{JPEG_QUALITY}"
    )
    print(f"每 {DETECT_PERIOD:.2f} 秒执行一次 NPU 推理")
    print(f"连续确认阈值：{CONFIRM_HIT_FRAMES} 次")
    print(f"全局报警冷却：{EVENT_COOLDOWN_SECONDS:.1f} 秒")
    print(f"CSV 兼容日志：{EVENT_LOG_PATH}")
    print(f"SQLite 数据库：{DATABASE_PATH}")
    print(f"自动留证目录：{EVIDENCE_DIR}")
    print(f"报警模式：{ALARM_MODE}")
    if ALARM_MODE == "led":
        print(
            f"LED GPIO：{LED_GPIO_CHIP} line {LED_GPIO_LINE}，"
            f"亮灯 {LED_ALARM_DURATION_SECONDS} 秒"
        )
    print(f"ROI 配置文件：{ROI_CONFIG_PATH}")
    print("入口线：支持 Web 页面拖动并与 ROI 一起保存")
    print("停止程序请按 Ctrl + C")

    try:
        app.run(
            host=WEB_HOST,
            port=WEB_PORT,
            threaded=WEB_THREADED,
            use_reloader=False
        )
    finally:
        running = False
        alarm.close()
        cap.release()
        rknn.release()
        print("LED、输入源和 RKNN 资源已经释放")
