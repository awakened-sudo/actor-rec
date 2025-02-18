# Actor Recognition System

A Python-based actor recognition system that uses deep learning to detect and identify actors in video frames. The system utilizes ResNet-50 for feature extraction and face recognition to match actors against reference images.

## Features

- Face detection and recognition using state-of-the-art deep learning models
- Support for multiple reference images per actor
- Scene-based processing with metadata generation
- Configurable similarity threshold for actor matching
- Detailed logging for debugging and analysis

## Project Structure

```
actor-rec/
├── actors/                    # Reference images of actors
│   ├── actor_p_ramlee.jpg
│   ├── actor_p_ramlee_2.jpg
│   └── actor_p_ramlee_3.jpg
├── temp_processing/          # Directory for processed frames
│   └── frames/              # Extracted video frames
├── actor.py                 # Main script
├── scene_metadata.json      # Input scene metadata
├── scene_metadata_with_cast.json  # Output with detected actors
└── requirements.txt         # Python dependencies
```

## Requirements

- Python 3.x
- PyTorch
- torchvision
- face_recognition
- PIL (Python Imaging Library)
- loguru

## Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your data:
   - Place reference images in the `actors/` directory
   - Name reference images as `actor_name.jpg` or `actor_name_N.jpg` for multiple images
   - Ensure scene frames are in `temp_processing/frames/`
   - Create a `scene_metadata.json` file with scene information

2. Run the script:
   ```bash
   python actor.py scene_metadata.json
   ```

3. Check results:
   - The script will generate `scene_metadata_with_cast.json`
   - Check the console output for detailed logging
   - Review detected actors and their confidence scores

## Scene Metadata Format

Input metadata (`scene_metadata.json`):
```json
{
    "scenes": [
        {
            "scene_start": "00:00:00",
            "scene_end": "00:00:10",
            "frames": {
                "description": "Scene description with actors",
                "sampled_frames": ["path/to/frame.jpg"]
            }
        }
    ]
}
```

## Configuration

Key parameters in `actor.py`:
- Similarity threshold: Adjust `threshold` in `compare_embeddings()` (default: 0.65)
- Person keywords: Modify `person_keywords` list in `process_scene()`
- Logging level: Configure through loguru settings

## Output

The system generates:
1. Console logging with detailed processing information
2. `scene_metadata_with_cast.json` containing detected actors per scene

## Troubleshooting

Common issues:
1. No faces detected: Ensure images are clear and faces are visible
2. Low confidence matches: Try adjusting the similarity threshold
3. Missing detections: Add more reference images from different angles

## License

This project is licensed under the MIT License - see the LICENSE file for details.
