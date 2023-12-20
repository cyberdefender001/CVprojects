import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_balance_time():
    cap = cv2.VideoCapture(0)
    balance_time = 0
    standing = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

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

                # Calculate balance time
                # abs(rightAnkle[0] - leftAnkle[0]) < 0.1 and
                if abs(rightAnkle[1] - leftAnkle[1]) < 0.1:
                    if not standing:
                        standing = True
                        balance_time += 1
                else:
                    standing = False

                cv2.putText(image, "Balance time: " + str(balance_time), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
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

    return balance_time


calculate_balance_time()