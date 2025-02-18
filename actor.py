import os
import json
import glob
import time
import logging
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import concurrent.futures
from datetime import timedelta
from pathlib import Path
from PIL import Image
import face_recognition
import torchvision.models as models
from torchvision import transforms
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Paths
APP_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
ACTORS_DIR = APP_DIR / "actors"
OUTPUT_DIR = APP_DIR / "video_analysis_output"

# Logging setup
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console)]
)
logger = logging.getLogger("ActorRecognition")

class ActorRecognizer:
    def __init__(self, num_workers=None):
        self.num_workers = num_workers or max(1, os.cpu_count() - 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.setup_model()
        self.reference_embeddings = self.load_reference_images()

    def setup_model(self):
        """Load ResNet-50 for feature extraction."""
        logger.info("Loading ResNet-50 model...")
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classifier layer
        model.to(self.device).eval()
        return model

    def extract_features(self, image_path):
        """Extract face embeddings using ResNet-50."""
        try:
            # Load and preprocess the image
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)

            # Detect faces
            face_locations = face_recognition.face_locations(image_np)
            logger.info(f"Found {len(face_locations)} faces in {image_path}")
            
            if not face_locations:
                logger.warning(f"No faces detected in {image_path}")
                return None  # Skip if no face detected

            # Get the first face location
            top, right, bottom, left = face_locations[0]
            face_image = image.crop((left, top, right, bottom))
            
            # Transform the face image
            face_tensor = transform(face_image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                embedding = self.model(face_tensor).squeeze()
                embedding = F.normalize(embedding, dim=0)  # Normalize for cosine similarity
            
            return embedding
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def load_reference_images(self):
        """Load reference images of actors and extract embeddings."""
        logger.info("Loading reference images...")
        reference_embeddings = {}

        for img_path in glob.glob(str(ACTORS_DIR / "actor_*.jpg")):
            actor_name = Path(img_path).stem.replace("actor_", "").replace("_", " ").title()
            embedding = self.extract_features(img_path)
            if embedding is not None:
                reference_embeddings[actor_name] = embedding
                logger.info(f"Loaded: {actor_name}")

        return reference_embeddings

    def compare_embeddings(self, embedding1, embedding2, threshold=0.65):
        """Compute cosine similarity between two embeddings."""
        similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
        logger.debug(f"Similarity score: {similarity:.4f}")
        return similarity > threshold

    def process_frame(self, frame_path):
        """Process a single frame and detect actors."""
        logger.info(f"Processing frame: {frame_path}")
        embedding = self.extract_features(frame_path)
        if embedding is None:
            logger.warning(f"No embedding generated for frame: {frame_path}")
            return []

        detected_actors = []
        for actor, ref_embedding in self.reference_embeddings.items():
            similarity = torch.nn.functional.cosine_similarity(embedding.unsqueeze(0), ref_embedding.unsqueeze(0)).item()
            logger.info(f"Similarity with {actor}: {similarity:.4f}")
            if similarity > 0.65:  # threshold
                detected_actors.append(actor)
                logger.info(f"Detected {actor} in frame {frame_path}")

        if not detected_actors:
            logger.warning(f"No actors matched in frame {frame_path}")
        return detected_actors

    def process_scene(self, scene_data):
        """Process a scene by analyzing sampled frames."""
        logger.info("Processing scene...")
        description = scene_data.get('frames', {}).get('description', "").lower()
        logger.info(f"Scene description: {description}")
        
        # More flexible keyword matching
        person_keywords = ['person', 'man', 'men', 'woman', 'people', 'child', 'colleague', 'companion', 'actor']
        if not any(keyword in description for keyword in person_keywords):
            logger.warning(f"Skipping scene - no person keywords ({', '.join(person_keywords)}) in description")
            return None

        detected_actors = set()
        frame_paths = scene_data['frames']['sampled_frames']
        logger.info(f"Processing {len(frame_paths)} frames in scene")
        
        for frame_path in frame_paths:
            logger.info(f"Processing frame path: {frame_path}")
            try:
                if not os.path.exists(frame_path):
                    logger.error(f"Frame file not found: {frame_path}")
                    continue
                    
                actors_in_frame = self.process_frame(frame_path)
                if actors_in_frame:
                    logger.info(f"Detected actors in frame: {actors_in_frame}")
                    detected_actors.update(actors_in_frame)
                else:
                    logger.warning(f"No actors detected in frame: {frame_path}")
                    
            except Exception as e:
                logger.error(f"Error processing frame {frame_path}: {str(e)}")
                continue

        if detected_actors:
            logger.info(f"Scene analysis complete. Detected actors: {detected_actors}")
            return {
                'scene_start': scene_data['scene_start'],
                'scene_end': scene_data['scene_end'],
                'cast': list(detected_actors)
            }
        else:
            logger.warning("No actors detected in any frames of the scene")
            return None

    def process_metadata_file(self, metadata_path):
        """Process a metadata file containing scene information."""
        try:
            with open(metadata_path, 'r') as f:
                scenes = json.load(f)
            
            modified = False
            with Progress(
                SpinnerColumn(), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing Scenes...", total=len(scenes))
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    future_to_scene = {executor.submit(self.process_scene, scene): i for i, scene in enumerate(scenes)}

                    for future in concurrent.futures.as_completed(future_to_scene):
                        scene_idx = future_to_scene[future]
                        try:
                            result = future.result()
                            if result:
                                scenes[scene_idx]['cast'] = result['cast']
                                modified = True
                                logger.info(f"Detected actors in scene {scene_idx}: {result['cast']}")
                        except Exception as e:
                            logger.error(f"Error processing scene {scene_idx}: {str(e)}")

                        progress.update(task, advance=1)

            if modified:
                output_path = metadata_path.replace('.json', '_with_cast.json')
                with open(output_path, 'w') as f:
                    json.dump(scenes, f, indent=2)
                logger.info(f"Updated metadata saved to: {output_path}")
            else:
                logger.info("No actors detected in any scenes")

        except Exception as e:
            logger.error(f"Failed to process metadata file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Detect actors in video scenes")
    parser.add_argument("metadata_file", help="Path to metadata JSON file")
    parser.add_argument("--workers", type=int, help="Number of worker processes (default: CPU count - 1)")
    args = parser.parse_args()

    if not os.path.exists(args.metadata_file):
        console.print(f"[red]Error: Metadata file not found: {args.metadata_file}")
        return

    start_time = time.time()
    try:
        recognizer = ActorRecognizer(num_workers=args.workers)
        recognizer.process_metadata_file(args.metadata_file)

        duration = time.time() - start_time
        console.print(f"[bold green]Processing Complete![/] Processing time: {timedelta(seconds=int(duration))}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")

if __name__ == "__main__":
    main()