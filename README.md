# Cross-Camera-Player-Mapping
hello my name is abdul muiz and this is my project.

This project maps player IDs between Tacticam and Broadcast videos using YOLOv11 for detection and color features for matching. It helps track the same player across different camera angles by comparing appearance features and outputting ID mappings for consistent multi-view analysis.
This project focuses on Cross-Camera Player Mapping, aiming to match player identities across two different camera views: Tacticam (top view) and Broadcast (side view). In multi-camera sports analytics, players often receive different IDs when tracked separately from each camera feed. This mismatch makes it difficult to combine or compare data across perspectives. To solve this, we use a trained YOLOv11 model to detect players in each video. From the detected bounding boxes, we extract simple color-based features — specifically, the mean RGB values — for each player. These features are saved as JSON files representing each player's appearance in both video feeds. Using this appearance data, we perform a matching process by calculating distances between feature vectors from the Tacticam and Broadcast videos. The closest matches are assumed to be the same player. The output is a mapping list such as: "Tacticam ID 1458 → Broadcast ID 1438", indicating that the player with ID 1458 in the Tacticam view corresponds to player 1438 in the Broadcast view. This mapping helps synchronize player data across camera feeds, which is important for video analysis, sports performance tracking, and tactical decision-making. The results can also be visualized in a tabular image format using matplotlib, making it easy to review and present the mapping. Overall, the project demonstrates an efficient and lightweight method to associate identities across multi-camera sports footage using feature extraction and similarity matching techniques.

Step-by-Step Process:
Load the Trained YOLO Model

It detects players, the ball, and referees in each frame.

Use Deep SORT for Tracking

Assigns a unique ID to each player so you can follow them across frames.

Load Precomputed Matches

Matches between Tacticam IDs and Broadcast IDs are loaded from a .json file (already generated).

Draw Tight Bounding Boxes

Each detected player is enclosed in a small green rectangle, not too big like default YOLO output.

Display Matched IDs

Each player’s label shows:
"player {Tacticam ID} → ID {Broadcast ID}"

Write Output Video

The annotated frames are saved into a new video file like
deepsort_matched_output_smallbox.mp4.

Output
 Matched player overlays on video

Compact boxes + minimal text

Great for presentations, GitHub, or reporting

