import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_air_time():
    cap = cv2.VideoCapture(0)
    air_time = 0
    start_time = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            # Check if frame is read correctly
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark

                # Get coordinates
                leftAnkle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                rightAnkle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate air time
                if leftAnkle[1] < 0.5 and rightAnkle[1] < 0.5:  # Assuming the ground is at y=0.5
                    if start_time is None:
                        start_time = time.time()
                else:
                    if start_time is not None:
                        air_time += time.time() - start_time
                        start_time = None

                cv2.putText(image, "Air time: {:.2f}".format(air_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 0, 255), 2)

            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    return air_time


print("Total air time: {:.2f} seconds".format(calculate_air_time()))