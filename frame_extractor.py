# frame_extractor.py

import cv2
import os

# --- Configuration ---
# 1. Path to the video you want to process
VIDEO_PATH = "examples/fire_video.mov"

# 2. The specific frame number you want to save (e.g., 150th frame)
#    Frame numbers start from 0.
FRAME_NUMBER_TO_SAVE = 50

# 3. The name of the output image file
OUTPUT_IMAGE_PATH = "saved_frame3.png"


def save_nth_frame(video_path, frame_number, output_path):
    """
    Opens a video file, seeks to the nth frame, and saves it as an image.

    Args:
        video_path (str): The full path to the video file.
        frame_number (int): The specific frame to save (0-indexed).
        output_path (str): The path to save the output image file.
    """
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at '{video_path}'")
        return

    # Open the video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file.")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if the requested frame number is valid
    if frame_number < 0 or frame_number >= total_frames:
        print(f"Error: Invalid frame number. The video has {total_frames} frames.")
        print(f"Please choose a frame number between 0 and {total_frames - 1}.")
        cap.release()
        return

    # Set the video capture position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame at that position
    success, frame = cap.read()

    if success:
        # If the frame was read successfully, save it to the specified path
        cv2.imwrite(output_path, frame)
        print(f"Successfully saved frame {frame_number} to '{output_path}'")
    else:
        print(f"Error: Failed to read frame {frame_number}.")

    # Release the video capture object
    cap.release()


if __name__ == '__main__':
    # Run the main function with the configured settings
    save_nth_frame(VIDEO_PATH, FRAME_NUMBER_TO_SAVE, OUTPUT_IMAGE_PATH)
