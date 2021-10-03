# SocialCam : Real-Time YOLOv4 Based Social Distancing Detector
YOLOv4-Based Real-Time Social Distancing Detector
<br/> A part of Kompetisi Penelitian Siswa Indonesia 2021 (KOPSI 2021)

## Author
<br/> By: Raihan Adhipratama Arvi
<br/> SMA Negeri 1 Sumatera Barat

## Feature
* CPU and CUDA Support
* Perspective Wrap Distance Measurement
* Integrated Calibration
* Beautiful and User-Friendly UI
* Argument Parser
* Configurable Parameter

### Usage

	usage: py SocialCam.py [arguments] [options]

	Available Arguments :

		-h, --help
					Show this message and exit
		-i, --input
					Input file location
					Default: configurable
		-c, --inputcam
					Connected webcam number ID
					Default: None
		-o, --output
					Output file name and directory
					Default: No output
		-g, --usegpu
					1 = use CUDA acceleration; 0 = use CPU
					Default: configurable
		-l, --log
					Output log file name
					Default: no output log
		-p, --getpoints
					1 = into perspective point calibration mode.
					Default: 0
		-a, --alarm
					1 = activate alarm; 0 = deactivate alarm.
					Default: condigurable
		-d, --display
					1 = display detection feed; 0 = no detection feed.
					Default: 1

### Requirement

Softwares requirement for SocialCam:
* Python 3.9.5
* OpenCV 4.5.2 w/ CUDA Support
* Numpy
