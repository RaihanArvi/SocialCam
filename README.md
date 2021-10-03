# SocialCam
YOLOv4-Based Real-Time Social Distancing Detector
A part of Kompetisi Penelitian Siswa Indonesia 2021 (KOPSI 2021)
By: Raihan Adhipratama Arvi
SMA Negeri 1 Sumatera Barat

## Feature
* DDD

## Usage

### Command Line

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

