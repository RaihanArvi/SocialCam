# SocialCam : Real-Time YOLOv4-Based Social Distancing Detector
YOLOv4-Based Real-Time Social Distancing Detector
<br/> SocialCam is a social distancing detector based on the YOLOv4 object detection algorithm. SocialCam uses perspective wrap to measure distance accurately. SocialCam outputs sound alarm when social distancing violation is detected. 

## Author
<br/> By: Raihan Adhipratama Arvi
<br/> SMA Negeri 1 Sumatera Barat

### Research Paper
This project is a part of Kompetisi Penelitian Siswa Indonesia 2021 (KOPSI 2021).
<br/> The research paper for this project can be found in the repository above.
<br/> Please consider citing the paper if you use this project in your research.

### Feature
* CPU and CUDA Support
* Perspective Wrap Distance Measurement
* Integrated Calibration
* Beautiful and User-Friendly UI
* Argument Parser
* Configurable Parameter

### Usage
SocialCam can be configured by editing the socialcam_config.py file.

#### Command Line Usage

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

### SocialCam Custom Model
SocialCam model download link: https://drive.google.com/drive/folders/1xsnl8BEHXpfZAIgPwPQcYQV8FGAQMkr2?usp=sharing

## Acknowledgement

Author would like to thank to all of the contributors who make this project possible. Especially Umi Nilma Herrita Wisda Syam, author's mentor in this project.
