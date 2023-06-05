
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from math import sqrt, atan2, degrees
import csv
import matplotlib.pyplot as plt

def select_pixel(event, x, y, flags, param):
    global selected_points, selected_count
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        selected_count += 1

def main():
    global selected_points, selected_count

    root = tk.Tk()
    root.withdraw()

    video_file = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
    if not video_file:
        print("No video file selected.")
        return
    
    cap = cv2.VideoCapture(video_file)
    _, first_frame = cap.read()

    selected_points = []
    selected_count = 0

    cv2.namedWindow("Select eight points: center, two rotating points, two horizontal movers, and two vertical movers")
    cv2.setMouseCallback("Select eight points: center, two rotating points, two horizontal movers, and two vertical movers", select_pixel)

    while selected_count < 7:
        cv2.imshow("Select eight points: center, two rotating points, two horizontal movers, and two vertical movers", first_frame)
        cv2.waitKey(20)

    center, rotating_point1, rotating_point2, mover_h1, mover_h2, mover_v1, mover_v2 = selected_points[0], selected_points[1], selected_points[2], selected_points[3], selected_points[4], selected_points[5], selected_points[6]
    cv2.destroyAllWindows()

    radius1 = sqrt((center[0] - rotating_point1[0])**2 + (center[1] - rotating_point1[1])**2)
    radius2 = sqrt((center[0] - rotating_point2[0])**2 + (center[1] - rotating_point2[1])**2)
    
    distances1, distances2, distances_h1, distances_h2, distances_v1, distances_v2, angle_changes1, angle_changes2 = [], [], [], [], [], [], [], []
    
    accumulated_distances1, accumulated_distances2 = [0], [0]
    accumulated_distances_h1, accumulated_distances_h2 = [0], [0]
    accumulated_distances_v1, accumulated_distances_v2 = [0], [0]
    
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev_point1 = np.array([[[float(rotating_point1[0]), float(rotating_point1[1])]]], dtype=np.float32)
    prev_point2 = np.array([[[float(rotating_point2[0]), float(rotating_point2[1])]]], dtype=np.float32)
    prev_point_h1 = np.array([[[float(mover_h1[0]), float(mover_h1[1])]]], dtype=np.float32)
    prev_point_h2 = np.array([[[float(mover_h2[0]), float(mover_h2[1])]]], dtype=np.float32)
    prev_point_v1 = np.array([[[float(mover_v1[0]), float(mover_v1[1])]]], dtype=np.float32)
    prev_point_v2 = np.array([[[float(mover_v2[0]), float(mover_v2[1])]]], dtype=np.float32)

    prev_angle1 = atan2(rotating_point1[1] - center[1], rotating_point1[0] - center[0])
    prev_angle2 = atan2(rotating_point2[1] - center[1], rotating_point2[0] - center[0])

    motion_history = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_point1, status1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point1, None)
        next_point2, status2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point2, None)
        next_point_h1, status_h1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point_h1, None)
        next_point_h2, status_h2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point_h2, None)
        next_point_v1, status_v1, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point_v1, None)
        next_point_v2, status_v2, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_point_v2, None)
        
        x11, y11 = prev_point1.ravel()
        x21, y21 = next_point1.ravel()
        x12, y12 = prev_point2.ravel()
        x22, y22 = next_point2.ravel()
        xh11, yh11 = prev_point_h1.ravel()
        xh21, yh21 = next_point_h1.ravel()
        xh12, yh12 = prev_point_h2.ravel()
        xh22, yh22 = next_point_h2.ravel()
        xv11, yv11 = prev_point_v1.ravel()
        xv21, yv21 = next_point_v1.ravel()
        xv12, yv12 = prev_point_v2.ravel()
        xv22, yv22 = next_point_v2.ravel()
        
        cv2.line(motion_history, (int(x11), int(y11)), (int(x21), int(y21)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(x12), int(y12)), (int(x22), int(y22)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(xh11), int(yh11)), (int(xh21), int(yh21)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(xh12), int(yh12)), (int(xh22), int(yh22)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(xv11), int(yv11)), (int(xv21), int(yv21)), (0, 255, 0), 2)
        cv2.line(motion_history, (int(xv12), int(yv12)), (int(xv22), int(yv22)), (0, 255, 0), 2)

        frame_with_history = cv2.add(frame, motion_history)

        

        cv2.circle(frame_with_history, center, 5, (255, 0, 0), -1)
        cv2.circle(frame_with_history, (int(x21), int(y21)), 5, (0, 255, 0), -1)
        cv2.circle(frame_with_history, (int(x22), int(y22)), 5, (0, 0, 255), -1)
        cv2.circle(frame_with_history, (int(xh21), int(yh21)), 5, (255, 0, 255), -1)
        cv2.circle(frame_with_history, (int(xh22), int(yh22)), 5, (255, 255, 0), -1)
        cv2.circle(frame_with_history, (int(xv21), int(yv21)), 5, (0, 255, 255), -1)
        cv2.circle(frame_with_history, (int(xv22), int(yv22)), 5, (255, 255, 255), -1)

        distance1 = sqrt((x21 - x11)**2 + (y21 - y11)**2)
        distance2 = sqrt((x22 - x12)**2 + (y22 - y12)**2)
        distance_h1 = sqrt((xh21 - xh11)**2 + (yh21 - yh11)**2)
        distance_h2 = sqrt((xh22 - xh12)**2 + (yh22 - yh12)**2)
        distance_v1 = sqrt((xv21 - xv11)**2 + (yv21 - yv11)**2)
        distance_v2 = sqrt((xv22 - xv12)**2 + (yv22 - yv12)**2)

        next_angle1 = atan2(y21 - center[1], x21 - center[0])
        next_angle2 = atan2(y22 - center[1], x22 - center[0])

        angle_change1 = degrees(next_angle1 - prev_angle1)
        angle_change2 = degrees(next_angle2 - prev_angle2)

        distances1.append(distance1)
        distances2.append(distance2)
        distances_h1.append(distance_h1)
        distances_h2.append(distance_h2)
        distances_v1.append(distance_v1)
        distances_v2.append(distance_v2)
        angle_changes1.append(angle_change1)
        angle_changes2.append(angle_change2)
        
        accumulated_distances1.append(accumulated_distances1[-1] + distance1)
        accumulated_distances2.append(accumulated_distances2[-1] + distance2)
        accumulated_distances_h1.append(accumulated_distances_h1[-1] + distance_h1)
        accumulated_distances_h2.append(accumulated_distances_h2[-1] + distance_h2)
        accumulated_distances_v1.append(accumulated_distances_v1[-1] + distance_v1)
        accumulated_distances_v2.append(accumulated_distances_v2[-1] + distance_v2)

        cv2.imshow("Motion tracking", frame_with_history)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray.copy()
        prev_point1 = next_point1.reshape(-1, 1, 2)
        prev_point2 = next_point2.reshape(-1, 1, 2)
        prev_point_h1 = next_point_h1.reshape(-1, 1, 2)
        prev_point_h2 = next_point_h2.reshape(-1, 1, 2)
        prev_point_v1 = next_point_v1.reshape(-1, 1, 2)
        prev_point_v2 = next_point_v2.reshape(-1, 1, 2)
        prev_angle1 = next_angle1
        prev_angle2 = next_angle2

    cap.release()
    cv2.destroyAllWindows()

 
    with open('motion_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Distance1", "Distance2", "AngleChange1", "AngleChange2", "Distance_h1", "Distance_h2", "Distance_v1", "Distance_v2"])
        for i in range(len(distances1)):
            writer.writerow([i, distances1[i], distances2[i], angle_changes1[i], angle_changes2[i], distances_h1[i], distances_h2[i], distances_v1[i], distances_v2[i]])

    plt.figure(figsize=(10, 10))

    plt.subplot(221)
    plt.title('Horizontal movers')
    plt.plot(distances_h1, label='Horizontal mover 1')
    plt.plot(distances_h2, label='Horizontal mover 2')
    plt.legend()

    plt.subplot(222)
    plt.title('Vertical movers')
    plt.plot(distances_v1, label='Vertical mover 1')
    plt.plot(distances_v2, label='Vertical mover 2')
    plt.legend()

    plt.subplot(223)
    plt.title('Rotating points distances')
    plt.plot(distances1, label='Rotating point 1')
    plt.plot(distances2, label='Rotating point 2')
    plt.legend()

    plt.subplot(224)
    plt.title('Rotating points angle changes')
    plt.plot(angle_changes1, label='Rotating point 1')
    plt.plot(angle_changes2, label='Rotating point 2')
    plt.legend()
    
    plt.subplot(325)
    plt.title('Accumulated displacements')
    plt.plot(accumulated_distances1, label='Rotating point 1')
    plt.plot(accumulated_distances2, label='Rotating point 2')
    plt.plot(accumulated_distances_h1, label='Horizontal mover 1')
    plt.plot(accumulated_distances_h2, label='Horizontal mover 2')
    plt.plot(accumulated_distances_v1, label='Vertical mover 1')
    plt.plot(accumulated_distances_v2, label='Vertical mover 2')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
