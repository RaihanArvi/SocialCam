############################################################## #
###
###  SocialCam Social Distancing Detection Program
###  By: Raihan Adhipratama Arvi
###  SMAN 1 Sumatera Barat
###  8 Oktober 2021
###
############################################################## #

# Packages
import cv2
import sys
import time
import random
import argparse
import datetime
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import socialcam_config as configuration
from PIL import Image
from math import sqrt
from imutils import paths

# ##########################################################
# ### FUNCTIONS
# ##########################################################

def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),2,(255,0,0),-1)
        ix,iy = x,y

def get_points(image,numOfPoints, image_size=(800,800)):
    global img
    img = image.copy()
    img = cv2.resize(img,image_size)
    width, height = image.shape[:2]
    cv2.namedWindow("image")
    cv2.setMouseCallback("image",draw_circle)
    points = []
    print("Press a for add point : ")
    while len(points) != numOfPoints:
        cv2.imshow("image",img)
        k = cv2.waitKey(1)
        if k == ord('a'):
            points.append([int(ix),int(iy)])
            cv2.circle(img,(ix,iy),3,(0,0,255),-1)
    cv2.destroyAllWindows()
    return np.float32(points)

def create_model(config, weights):

    model = cv2.dnn.readNetFromDarknet(config, weights)

    if USE_GPU == 1:
        print('Using GPU')
        print('Setting CUDA Backend')
        backend = cv2.dnn.DNN_BACKEND_CUDA
        target = cv2.dnn.DNN_TARGET_CUDA
    else:
        print('Using CPU')
        backend = cv2.dnn.DNN_BACKEND_OPENCV
        target = cv2.dnn.DNN_TARGET_CPU

    model.setPreferableBackend(backend)
    model.setPreferableTarget(target)
    return model

def get_output_layers(model):

    layer_names = model.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in model.getUnconnectedOutLayers()]
    return output_layers

def blob_from_image(image, target_size):

    blob = cv2.dnn.blobFromImage(image, 1/255., target_size, [0,0,0], 1, crop=False)
    return blob

def predict(blob, model, output_layers):

    model.setInput(blob)
    outputs = model.forward(output_layers)
    return outputs

def get_image_boxes(outputs, image_width, image_height, classes, confidence_threshold=0.5, nms_threshold=0.4):
    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            class_name = classes[class_id]
            confidence = scores[class_id]
            if confidence > confidence_threshold and class_name== 'person':
                cx, cy, width, height = (detection[0:4] * np.array([image_width, image_height, image_width, image_height])).astype("int")
                x = int(cx - width / 2)
                y = int(cy - height / 2)
                boxes.append([x, y, int(width), int(height),cx,cy])
                confidences.append(float(confidence))
    nms_indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    nms_indices = np.array(nms_indices)
    return [boxes[ind] for ind in nms_indices.flatten()]

def compute_point_perspective_transformation(matrix,boxes):
    list_downoids = [[box[4], box[5]+box[3]//2] for box in boxes]
    list_points_to_detect = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points_to_detect, matrix)
    transformed_points_list = list()
    if transformed_points is not None:
        for i in range(0,transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0],transformed_points[i][0][1]])
    return np.array(transformed_points_list).astype('int')

def get_red_green_boxes(distance_allowed,birds_eye_points,boxes):
    red_boxes = []
    green_boxes = []

    distanceOutput = [] # untuk pengujian jarak

    new_boxes = [tuple(box) + tuple(result) for box, result in zip(boxes, birds_eye_points)]

    distCount = 0

    for i in range(0, len(new_boxes)-1):
            for j in range(i+1, len(new_boxes)):
                cxi,cyi = new_boxes[i][6:]
                cxj,cyj = new_boxes[j][6:]
                distance = eucledian_distance([cxi,cyi], [cxj,cyj])
                distround = round(distance, 2)

                distanceOutput.append(distround)
                distCount =+ 1

                if distance < distance_allowed:
                    red_boxes.append(new_boxes[i])
                    red_boxes.append(new_boxes[j])

    green_boxes = list(set(new_boxes) - set(red_boxes))
    red_boxes = list(set(red_boxes))

    return (green_boxes, red_boxes, distanceOutput)

def eucledian_distance(point1, point2):
    x1,y1 = point1
    x2,y2 = point2
    return sqrt((x1-x2)**2 + (y1-y2)**2)

def get_birds_eye_view_image(green_box, red_box):
    templateBgr = cv2.imread('(2) Resources/(1) UI/warpedBgr.jpg')
    compValY = 0 # Atur Ketinggian Titik-Titik
    compValX = 0

    cv2.putText(templateBgr, str(len(red_box)), (120,82), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,0,255), 4, cv2.LINE_AA)
    cv2.putText(templateBgr, str(len(green_box)), (425,82), cv2.FONT_HERSHEY_SIMPLEX , 2, (0,255,0), 4, cv2.LINE_AA)

    for point in green_box:
        cv2.circle(templateBgr,tuple([point[6]+compValX,point[7]+compValY]),15,(0,255,0),-1)
    for point in red_box:
        cv2.circle(templateBgr,tuple([point[6]+compValX,point[7]+compValY]),15,(0,0,255),-1)
    return templateBgr

def get_red_green_box_image(new_box_image,green_box,red_box):
    for point in green_box:
        cv2.rectangle(new_box_image,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(0, 255, 0), 2)
    for point in red_box:
        cv2.rectangle(new_box_image,(point[0],point[1]),(point[0]+point[2],point[1]+point[3]),(0, 0, 255), 2)
    return new_box_image

def alarmRing():
    audioPath = configuration.AUDIO_PATH
    audioFile = configuration.AUDIO_FILE_NAMES
    programSound = r'(1) Required\(3) cmdmp3win\cmdmp3win'

    commandCMD = programSound + ' ' + audioPath + audioFile[random.randint(0, len(audioFile))]

    subprocess.Popen(commandCMD)
    print('Alarm Triggered')

def checkAlarm(detected, timerState, startTimer, pakaiAlarm, alarmRinging):

    if pakaiAlarm > 0:
        if detected > violationsThresh and timerState == False:
            timerState = True
            startTimer = time.time()
        elif detected < violationsThresh and timerState == True:
            timerState = False

        elif detected > violationsThresh and timerState == True and alarmRinging == False:
            print('Start Timer =')
            print(startTimer)

            stopTimer = time.time()
            interval = stopTimer - startTimer
            print(interval)
            if interval > timerThresh:
                alarmRing()
                alarmRinging = True
                startTimer = time.time()
                timerState = False

    return startTimer, timerState, alarmRinging

# ##########################################################
# ### Argparse
# ##########################################################

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default=configuration.DEFAULT_INPUT,
	help="lokasi file input")
ap.add_argument("-c", "--inputcam", type=int, default=None,
	help="lokasi file input / nomor webcam (optional)")
ap.add_argument("-o", "--output", type=str, default="",
	help="lokasi video output (optional). kosong = no output")
ap.add_argument("-g", "--usegpu", type=int, default=configuration.USE_GPU,
	help="1 = pake gpu. 0 = pake cpu. default di config file")
ap.add_argument("-l", "--log", type=str, default="",
	help="nama file log output (optional)")
ap.add_argument("-p", "--getpoints", type=int, default=0,
	help="1 = ambil point perspektif dari video (optional)")
ap.add_argument("-a", "--alarm", type=int, default=configuration.DEFAULT_ALARM,
	help="1 hidupkan alarm. 0 matikan alarm.")
ap.add_argument("-d", "--display", type=int, default=1,
	help="(1) tampilkan feed / tidak (0)")
args = vars(ap.parse_args())

# File Locations Variables #

img = []
ix,iy = 0,0

USE_GPU = args['usegpu']

if args['inputcam'] is not None:
    inputVideo = args['inputcam']
else:
    inputVideo = args['input']

welcomeScreen = cv2.imread('(2) Resources/(1) UI/welcome_screen.jpg') #Splash Screen
main_header = cv2.imread('(2) Resources/(1) UI/top.jpg')
bot_header = cv2.imread('(2) Resources/(1) UI/bottom.jpg')
overlayFile = cv2.imread('(2) Resources/(1) UI/overlay.png', cv2.IMREAD_UNCHANGED)
(wH, wW) = overlayFile.shape[:2]

if args["display"] > 0:
    cv2.namedWindow('Welcome to SocialCam', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Welcome to SocialCam', cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Welcome to SocialCam', int(1920/2), int(1080/2))
    cv2.imshow('Welcome to SocialCam', welcomeScreen)
    cv2.waitKey(3000)
    cv2.destroyWindow('Welcome to SocialCam')

# OpenCV Bug (Leave it as is)
(B, G, R, A) = cv2.split(overlayFile)
B = cv2.bitwise_and(B, B, mask=A)
G = cv2.bitwise_and(G, G, mask=A)
R = cv2.bitwise_and(R, R, mask=A)
overlayFile = cv2.merge([B, G, R, A])
# ######################### #

# ##########################################################
# ### Perspective Source Points
# ##########################################################

if args['getpoints'] > 0:
    print(' ')
    print('--- Calibration Process...')
    done = True
    videoSource = inputVideo
    framePath = configuration.FRAME_PATH
    frameSeq = configuration.FRAME_SEQ
    getFrame = cv2.VideoCapture(videoSource)
    frameCount = 0

    print('Mengekstrak Frame dari '+str(videoSource))
    print('Ke '+str(framePath))
    print('Frame yang diekstrak : '+str(frameSeq))
    print('Done')
    print('.')
    print(' ')

    print('--- Extracting...')
    while done:
        ret, frame = getFrame.read()

        if not ret:
            break

        frameCount += 1
        if frameCount in frameSeq:
            print("Frame number : ", frameCount, "Extracted")
            cv2.imwrite(framePath.format(frameCount), frame)
        if frameCount == frameSeq[len(frameSeq)-1]:
            print('Done')
            print('.')
            done = False
            break
    getFrame.release()

    srcPoints = input('Masukkan Nama Frame Sumber dalam Folder (5) Frames/ : ')
    print('.')
    srcPoints = '(5) Frames/' + srcPoints

    srcFrame = cv2.imread(srcPoints)
    frameWidth = srcFrame.shape[1]
    frameHeight = srcFrame.shape[0]

    srcFrameCopy = srcFrame.copy()
    source_points = get_points(srcFrameCopy, 4, image_size=(frameWidth, frameHeight))
    print('')
    print('.')
    print('Titik Perspektif = '+str(source_points))

    filePers = open('perspective_points_'+str(datetime.date.today())+'.txt', "w+")
    filePers.write('Titik Perspektif : \n')
    filePers.write(str(source_points))
    filePers.close()

    for point in source_points:
        #print(tuple(point))
        cv2.circle(srcFrameCopy, (142, 298), 8, (255, 0, 0), -1)

    points = source_points.reshape((-1, 1, 2)).astype(np.int32)

    cv2.polylines(srcFrameCopy, [points], True, (0, 255, 0), thickness=4)

    cv2.imwrite('(5) Frames/Perspective Points.jpg', srcFrameCopy)
    print('.')
    print('Writing Points Perspective Image to ./(5) Frames/Perspective Points.jpg')

    cv2.imshow('Frames', srcFrameCopy)
    cv2.waitKey(2500)
    cv2.destroyWindow('Frames')

else:
    source_points = np.float32(configuration.SOURCE_POINTS)
    points = source_points.reshape((-1, 1, 2)).astype(np.int32)

# ##########################################################
# ### Points Scale Value
# ##########################################################

src = source_points
dst = np.float32(configuration.DST)

dst_size = (582, 753)
dst = dst * np.float32(dst_size) # Turn off Jika Ingin Pakai Langsung (Refer ke cv2.getPerspectiveTransform)

H_matrix = cv2.getPerspectiveTransform(src, dst)
# ##########################################################
# ### Warped Image Preview
# ##########################################################

if args['getpoints'] > 0:
    warpedEx = cv2.warpPerspective(srcFrameCopy, H_matrix, dst_size)
    cv2.imwrite('(5) Frames/Warped Image.jpg', warpedEx)
    print('Writing Warped Images to ./(5) Frames/Warped Image.jpg')
    print('.')

    cv2.imshow('Warped Frame', warpedEx)
    cv2.waitKey(2500)
    cv2.destroyWindow('Warped Frame')


# ##########################################################
# ### Variable
# ##########################################################

# Basic Configuration
confidence_threshold = configuration.MIN_CONFIDENCE
nms_threshold = configuration.NMS_THRESH
min_distance = configuration.MIN_DISTANCE

# Alarm Configuration
violationsThresh = configuration.VIOL_THRESH
timerThresh = configuration.TIMER_THRESH

# Blob Target Size
width = configuration.BLOB_TARGET_WIDTH
height = configuration.BLOB_TARGET_HEIGHT

# YOLO Weight and Classes
classes = configuration.CLASSES
config = configuration.CONFIG
weights = configuration.WEIGHTS

with open(classes, 'rt') as f:
    coco_classes = f.read().strip('\n').split('\n')

model = create_model(config, weights)
output_layers = get_output_layers(model)

# ##########################################################
# ### Main Loop
# ##########################################################

video = cv2.VideoCapture(inputVideo)
fullscreenState = False

# Alarm Configuration
timerState = False
startTimer = 0
alarmIsRinging = False

# For Video Export
writer = None
frame_number = 1

if args["log"] != "":
    logState = True
else:
    logState = False

if logState == True:
    log = open('fps_'+args['log']+str(datetime.date.today()), "w+")
    log2 = open('detection_' + args['log']+str(datetime.date.today()), "w+")
    log3 = open('fpsProc_'+args['log']+str(datetime.date.today()), "w+")

    startTime = time.time() # Log Lama Keseluruhan Process

    log.write('Start FPS Logging\n')
    log.write('USE GPU : '+str(USE_GPU)+'\n')
    log.write('Start Date Time : '+str(datetime.date.today())+'\n')
    log.write('Start Time : '+str(startTime)+'\n')
    log.write('\n')

    log2.write('Start Detection Logging\n')
    log2.write('USE GPU : ' + str(USE_GPU) + '\n')
    log2.write('Processing Frame'+'|Total Detected Person'+'|Red Marked Person'+'|Green Marked Person\n')

    log3.write('Start FPS Processing Logging\n')
    log3.write('USE GPU : ' + str(USE_GPU) + '\n')
    log3.write('Start Date Time : ' + str(datetime.date.today()) + '\n')
    log3.write('Start Time : ' + str(startTime) + '\n')
    log3.write('\n')

if args["output"] != "" and writer is None:
    print('Processing Frame'+'|Total Detected Person'+'|Red Markerd Person'+'|Green Marked Person\n')

while True:

    startFPS = time.time()
    ret, frame = video.read()

    if not ret:
        break

    image_height, image_width = frame.shape[:2]

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    blob = blob_from_image(image, (width, height))
    outputs = predict(blob, model, output_layers)

    endFPS = time.time()

    boxes = get_image_boxes(outputs, image_width, image_height, coco_classes)
    birds_eye_points = compute_point_perspective_transformation(H_matrix, boxes)
    green_box, red_box, outputDistance = get_red_green_boxes(min_distance, birds_eye_points, boxes)
    birds_eye_view_image = get_birds_eye_view_image(green_box, red_box)
    box_red_green_image = get_red_green_box_image(frame.copy(), green_box, red_box)

    box_red_green_image = cv2.resize(box_red_green_image, (1920, 1080))

    (h, w) = box_red_green_image.shape[:2]
    image = np.dstack([box_red_green_image, np.ones((h, w), dtype="uint8") * 255])
    overlay = np.zeros((h, w, 4), dtype="uint8")
    overlay[0:h, 0:w] = overlayFile

    outputs = image.copy()
    cv2.addWeighted(overlay, 0.65, outputs, 1.0, 0, outputs)
    outputs = outputs[:, :, :3]
    outputs = outputs.copy()

    intervalFPS = endFPS - startFPS
    fps = 1 / intervalFPS

    startTimer, timerState, alarmIsRinging = checkAlarm(len(red_box), timerState, startTimer, args['alarm'], alarmIsRinging)

    if alarmIsRinging == True:
        alarmStart = startTimer
        alarmStop = time.time()
        alarmHasRingInt = alarmStop - alarmStart
        cv2.putText(outputs, "Memberikan Peringatan", (60, h - 115), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        if alarmHasRingInt > configuration.TIMER_ELAPSED:
            alarmIsRinging = False

    humTotal = len(red_box) + len(green_box)
    cv2.putText(outputs, "Total Manusia Terdeteksi : "+str(humTotal), (60, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(outputs, str(format(fps, ".2f")) + " FPS", (w-300, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(outputs, "Safe Distance : " + str(min_distance) + " px", (w - 500, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(outputs, str(datetime.date.today()), (w - 320, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
    cv2.putText(outputs, str(datetime.datetime.now().strftime("%H:%M")), (w - 160, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

    if USE_GPU == 1:
        cv2.putText(outputs, "GPU/CUDA", (w - 300, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        cv2.putText(outputs, "ON CPU", (w - 240, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    cv2.putText(outputs, "RISIKO TINGGI : "+str(len(red_box))+" orang", (140, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(outputs, "RISIKO RENDAH : " + str(len(green_box)) + " orang", (140, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    resized = cv2.resize(outputs, (1338, 753))

    combined_image = np.concatenate((birds_eye_view_image, resized), axis=1)
    deshboard_image = np.concatenate((main_header, combined_image), axis=0)
    deshboard_image = np.concatenate((deshboard_image, bot_header), axis=0)

    endFPSProcess = time.time()
    intervalFPSProcess = endFPSProcess - startFPS
    fpsProcess = 1 / intervalFPSProcess

    if logState == True:
        log.write(str(frame_number)+','+str(format(fps, ".2f"))+'\n')
        log2.write('%-20i|%-25i|%-25i|%-25i\n' % (frame_number, len(boxes), len(red_box), len(green_box)))
        log3.write(str(frame_number) + ',' + str(format(fpsProcess, ".2f")) + '\n')

    if args["output"] != "" and writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter(args['output'], fourcc, 15,
                                 (deshboard_image.shape[1], deshboard_image.shape[0]), True)

    if writer is not None:
        writer.write(deshboard_image)
        print('%-20i|%-25i|%-25i|%-25i\n' % (frame_number, len(boxes), len(red_box), len(green_box)))

    frame_number = frame_number + 1

    key = cv2.waitKey(1) & 0xFF

    if args["display"] > 0:
        if key == ord("f"):
            fullscreenState = not fullscreenState
        if fullscreenState == True:
            cv2.namedWindow('SocialCam', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('SocialCam', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow('SocialCam', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('SocialCam', cv2.WINDOW_NORMAL, cv2.WINDOW_NORMAL)
            cv2.resizeWindow('SocialCam', int(1920/2), int(1080/2))

        cv2.imshow('SocialCam', deshboard_image)
        if key == 27:
            cv2.destroyAllWindows()
            break

    del image, outputs, combined_image, deshboard_image, birds_eye_view_image

print(' ')
if logState == True:
    endTime = time.time()
    intrvl = endTime - startTime
    log.write('\n')
    log.write('End Date Time : ' + str(datetime.now()) + '\n')
    log.write('End Time : '+endTime+'\n')
    log.write('Interval : ' + intrvl + '\n')
    log.write('Total Frames : '+frame_number+'\n')
    log.write('Finished\n')
    log.close()

    log2.write('\n')
    log2.write('End Date Time : ' + str(datetime.now()) + '\n')
    log2.write('End Time : ' + endTime + '\n')
    log2.write('Interval : ' + intrvl + '\n')
    log2.write('Total Frames : ' + frame_number + '\n')
    log2.write('Finished\n')
    log2.close()

    log3.write('\n')
    log3.write('End Date Time : ' + str(datetime.now()) + '\n')
    log3.write('End Time : ' + endTime + '\n')
    log3.write('Interval : ' + intrvl + '\n')
    log3.write('Total Frames : ' + frame_number + '\n')
    log3.write('Finished\n')
    log3.close()

if args["output"] != "":
    writer.release()

video.release()
