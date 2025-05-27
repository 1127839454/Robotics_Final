# Ball Detection Project

## Quick Start

### 1. Install Dependencies
First, install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Application
Execute the main script to find the ball:
```bash
python main.py
```

## Project Structure

```
├── model/                    # Model files and weights
├── Robotic/                  # Robotic control modules
├── Server/                   # Server-related components
├── yolov5/                   # YOLOv5 detection framework
│   ├── detect.py            # Object detection script
│   ├── train.py             # Model training script
│   ├── val.py               # Model validation script
│   ├── export.py            # Model export utilities
│   ├── voice_video_recognition.py  # Voice and video recognition code
│   ├── benchmarks.py        # Performance benchmarking
│   └── ...                  # Other YOLOv5 utilities
├── main.py                   # Main application entry point
├── command.txt              # Command configurations
├── params.json              # Parameter settings
├── output.h264              # Video output file
└── requirements.txt         # Python dependencies
```

## Key Components

- **main.py**: The script for controlling the car's operation
- **yolov5/**: Contains the YOLOv5 object detection framework with additional voice and video recognition capabilities
- **voice_video_recognition.py**: Located in the yolov5 directory, this module handles voice and video recognition functionality
- **model/**: Stores trained model weights and configurations
- **Robotic/**: Contains robotic control and movement logic
- **Server/**: Server components for handling requests and responses

## Configuration

Modify `params.json` to adjust detection parameters and settings according to your needs.