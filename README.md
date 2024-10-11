# Person Detection and Tracking Project Report

## Approach and Model Selection

For this project, I implemented a person detection and tracking system using computer vision techniques. The primary goal was to accurately detect people in video frames and track their movements over time.

### Model Selection

I experimented with several state-of-the-art object detection models from the YOLO (You Only Look Once) family, including YOLOv5, YOLOv7, and YOLOv11. After extensive testing, I found that YOLOv5s provided the best balance of accuracy and performance for our specific use case. Here's why:

1. **Speed**: YOLOv5s is relatively lightweight, allowing for real-time processing of video frames.
2. **Accuracy**: Despite being smaller than its counterparts, YOLOv5s demonstrated high accuracy in person detection tasks.
3. **Ease of Use**: The PyTorch implementation of YOLOv5 is well-documented and easy to integrate into our pipeline.
4. **Community Support**: YOLOv5 has a large user base, which means better resources and ongoing improvements.

### Tracking Algorithm

For object tracking, I implemented the Norfair tracking library, which uses a simple yet effective algorithm based on Euclidean distance. This choice was made because:

1. It's lightweight and fast, complementing the real-time capabilities of YOLOv5s.
2. It handles object ID assignment and maintenance effectively.
3. It's easy to integrate with our detection pipeline.

## Challenges and Solutions

Throughout the development process, I encountered several challenges:

1. **Codec Compatibility**: Some video codecs were not compatible with OpenCV's VideoWriter.

   Solution: I standardized the output to MP4 format using the 'mp4v' codec, which has broad compatibility.

1. **Inconsistent Bounding Boxes**: Initially, the system was assigning IDs to detected persons without drawing bounding boxes around them, which made it difficult to visually associate IDs with specific individuals.

   Solution: I modified the detection and visualization code to ensure that a bounding box is drawn for each detected person before assigning and displaying the ID. This significantly improved the visual clarity of the output.

## Instructions for Use

The code for this project is available on GitHub [insert your GitHub link here]. To use this code:

1. Open the notebook in Google Colab.
2. Change the runtime type to T4 GPU or any available GPU option. This can be done by going to Runtime > Change runtime type > Hardware accelerator > GPU.
3. Run the initial cells to install necessary dependencies.
4. Choose your video input method:
   - For YouTube videos: Uncomment the YouTube download section and comment out the file upload section.
   - For local video upload: Keep the file upload section uncommented and the YouTube section commented out.
5. Run the remaining cells to process the video and generate the output.

Note: Ensure you have sufficient Colab runtime allocated, especially for longer videos. Using a GPU runtime is crucial for efficient processing of video frames.

## Future Improvements

While the current implementation is functional, there are several areas for potential improvement:

1. Implement multi-class object detection for more diverse tracking scenarios.
2. Explore more advanced tracking algorithms for improved accuracy in crowded scenes.
3. Add options for adjusting detection confidence thresholds and tracking parameters.
4. Implement a user-friendly interface for easier parameter tuning and video selection.
5. Optimize the code to handle longer videos more efficiently, possibly by implementing frame skipping or more advanced memory management techniques.

By continually refining and expanding this system, we can create an even more robust and versatile tool for person detection and tracking tasks.
