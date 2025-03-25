# Hand Gesture Volume Control

A computer vision-based system that uses hand gestures to control system volume, featuring MediaPipe for hand gesture recognition and face detection.

![Hand Gesture Volume Control Demo](https://github.com/wangqiqi/interesting_assets/raw/main/images/hand_vol.png)

## Features

- Real-time hand gesture recognition
- Face detection and masking
- Smooth volume control
- Visual volume display
- Real-time FPS counter
- Privacy protection with face masking

## Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- pycaw
- numpy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hand_control
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main program:
```bash
python hand_control_volume.py
```

## Controls

- Use the distance between your thumb and index finger to control volume
- Increase distance to increase volume
- Decrease distance to decrease volume
- Press ESC to exit

## Configuration

You can adjust the following parameters in the `HandControlVolume` class `__init__` method:

- `face_mask_color`: Color of the face mask
- `face_mask_alpha`: Face mask transparency
- `gesture_threshold`: Hand gesture detection threshold
- `smooth_factor`: Volume smoothing factor

## Privacy Features

- Automatic face detection and masking
- Configurable mask color and transparency
- Real-time face tracking

## Contact

If you have any questions or suggestions, feel free to contact me:

- WeChat: znzatop

![WeChat](https://github.com/wangqiqi/interesting_assets/raw/main/images/wechat.jpg)

## More Projects

更多有趣的项目请见：https://github.com/wangqiqi/interesting_assets.git

## License

MIT License 