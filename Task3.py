import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Parameters for motion tracking
path_points = deque(maxlen=500)  # Saves the detected body's movement path
background_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

def remove_human_body(frame, bg_subtractor):
    
    fg_mask = bg_subtractor.apply(frame)

    # reduce noise in the foreground mask
    fg_mask = cv2.medianBlur(fg_mask, 5)

    # Threshold the mask to get a binary image
    _, fg_mask = cv2.threshold(fg_mask, 240, 255, cv2.THRESH_BINARY)

    # operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours of moving areas
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Filter small areas
            # Draw black over the detected body area to "remove" it
            x, y, w, h = cv2.boundingRect(largest_contour)
            frame[y:y+h, x:x+w] = 0  # Set detected area to black

            # Track the center point for the path
            centerX, centerY = x + w // 2, y + h // 2
            path_points.append((centerX, centerY))

    return frame

def display_histogram(frame):
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title("Histogram for First Frame")
    plt.xlabel("Bins")
    plt.ylabel("Number of Pixels")
    
    for i, color in enumerate(colors):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.show()

def display_path(frame, path_points):
    for i in range(1, len(path_points)):
        if path_points[i - 1] is None or path_points[i] is None:
            continue
        cv2.line(frame, path_points[i - 1], path_points[i], (0, 255, 0), 2)

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    first_frame_shown = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if not first_frame_shown:
            display_histogram(frame)
            first_frame_shown = True

        processed_frame = remove_human_body(frame, background_subtractor)
        display_path(processed_frame, path_points)

        # Display the processed frame
        cv2.imshow("Processed Frame", processed_frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Plot the human body movement path
    if path_points:
        x_points, y_points = zip(*path_points)
        plt.figure()
        plt.plot(x_points, y_points, 'o-', markersize=3, color='red')
        plt.title("Human Body Movement Path")
        plt.xlabel("X Coordinates")
        plt.ylabel("Y Coordinates")
        plt.show()

if __name__ == "__main__":
    video_path = 'task3.mp4'  
    main(video_path)
