#
# SocialCam Config File
# 05 July 2021
#

# Lokasi Weight YOLO
CLASSES = '(1) Required/(2) Names/coco.names'
# ------------------
CONFIG = '(1) Required/(1) Models/YOLOv4_tiny/yolov4-tiny.cfg'
WEIGHTS = '(1) Required/(1) Models/YOLOv4_tiny/yolov4-tiny.weights'

# Input Default (Bisa File Video atau Kamera Webcam)
DEFAULT_INPUT = '(3) Input/720p.mp4'

# Frame Untuk Kalibrasi Perspektif
FRAME_PATH = '(5) Frames/CALIBRATION_FRAME_{}.jpg' # Output
FRAME_SEQ = [100, 200, 300, 400, 500]
#FRAME_SEQ = [5, 10, 15, 20, 25] # For Real Time Camera

# Pakai NVIDIA CUDA GPU Secara Default (1) atau CPU Default (0)
USE_GPU = 1

# Jarak Minimal Antar Manusia Terdeteksi (dalam piksel)
MIN_DISTANCE = 80

# Alarm
DEFAULT_ALARM = 1 # 0 = Tanpa Alarm; 1 = Pakai Alarm
VIOL_THRESH = 2 # Minimal Pelanggaran Untuk Pengaktifan Alarm
TIMER_THRESH = 5 # Waktu Interval SETELAH pelanggaran terdeteksi sebanyak [violationThresh] untuk
                # mengaktifkan alarm

TIMER_ELAPSED = 10 # Lama alarm dormant

#######################################################

# Audio Files List
AUDIO_PATH = r'(2) Resources\(2) Sound'
AUDIO_FILE_NAMES = ['1.mp3', '2.mp3', '3.mp3', '4.mp3', '5.mp3']

MIN_CONFIDENCE = 0.5
NMS_THRESH = 0.4

# Blob Target Size
BLOB_TARGET_WIDTH = 608
BLOB_TARGET_HEIGHT = 608

# ################# #
# Perspektif Points
# ################# #


SOURCE_POINTS = [[ 796.,  180.],
                             [1518.,  282.],
                             [1080.,  719.],
                             [ 128.,  480.]]

DST = [(0.1,0.5), (0.69, 0.5), (0.69,0.8), (0.1,0.8)]

