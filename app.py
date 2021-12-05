import cv2
import mediapipe as mp
import math
import statistics

video_capture = cv2.VideoCapture(0)
mediapipe = mp.solutions.hands
mediapipe_hands = mediapipe.Hands(static_image_mode= False, max_num_hands = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

def softmax_translation(softmax_argmax):
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    translation = alphabet[softmax_argmax]
    return translation

def calculate_hand_centroid_velocity(centroid_position_1_x_y, centroid_position_2_x_y):
    distance_between_pos_1_2 = math.sqrt((math.pow(centroid_position_2_x_y[0] - centroid_position_1_x_y[0], 2)) + (math.pow(centroid_position_2_x_y[1] - centroid_position_1_x_y[1], 2)))
    time_velocity = float(1)
    hand_centroid_velocity = distance_between_pos_1_2 / time_velocity
    return hand_centroid_velocity

def calculate_hand_anchor_distances(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance   

while video_capture:
    _, original_frame = video_capture.read()
    frame_y_max = original_frame.shape[0]
    frame_x_max = original_frame.shape[1]   
    analytics_frame = original_frame.copy()
    test_frame = original_frame.copy()
    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    mediapipe_hands_results = mediapipe_hands.process(original_frame_rgb)
    hand_landmark_x_coordinates = []
    hand_landmark_y_coordinates = []
    
    if mediapipe_hands_results.multi_hand_landmarks:
        for hand_landmarks in mediapipe_hands_results.multi_hand_landmarks:            
            landmark_points_color = mp_drawing.DrawingSpec(color=(221,0,255), thickness=4, circle_radius=1)
            landmark_connections_points_color = mp_drawing.DrawingSpec(color=(221,0,255), thickness=4, circle_radius=1)
            mp_drawing.draw_landmarks(test_frame, hand_landmarks, mediapipe.HAND_CONNECTIONS, landmark_points_color, landmark_connections_points_color)             

            for id, lm in enumerate(hand_landmarks.landmark):                
                h, w, c = original_frame.shape  
                hand_landmark_coordinate_x, hand_landmark_coordinate_y = int(lm.x * w), int(lm.y * h)                                      
                hand_landmark_x_coordinates.append(hand_landmark_coordinate_x)
                hand_landmark_y_coordinates.append(hand_landmark_coordinate_y)                      

        hand_bounding_box_x_min = min(hand_landmark_x_coordinates)
        hand_bounding_box_x_max = max(hand_landmark_x_coordinates)
        hand_bounding_box_y_min = min(hand_landmark_y_coordinates)
        hand_bounding_box_y_max = max(hand_landmark_y_coordinates)          

        # BOUNDING BOX AND CENTROID
        cv2.rectangle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 255, 0), 2)            
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 1)                                    

        # BOUNDING BOX ANCHORS
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), 5, (255, 0, 0), 10)
        top_middle_anchor_distance = calculate_hand_anchor_distances([int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2),int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)])
        
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), 5, (255, 0, 0), 10)     
        bottom_middle_anchor_distance = calculate_hand_anchor_distances([int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)], [int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)])

        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (255, 0, 0), 10)            
        left_middle_anchor_distance = calculate_hand_anchor_distances([int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)])

        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (255, 0, 0), 10)            
        right_middle_anchor_distance = calculate_hand_anchor_distances([int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)])                        

        anchor_distances_list = [top_middle_anchor_distance, bottom_middle_anchor_distance, left_middle_anchor_distance, right_middle_anchor_distance]            

        anchor_distances_avg = float(statistics.mean(anchor_distances_list))

        # HAND POSITION EVALUATION
        if hand_bounding_box_x_min < (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150/1.25):
            hand_position_eval = "TOO_CLOSE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)                 
        elif hand_bounding_box_x_min > (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150/1.25) and anchor_distances_avg < 70:                
            hand_position_eval= "IN_RANGE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 255, 0), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 255, 0), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 255, 0), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 255, 0), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 255, 0), 3)                  
        elif hand_bounding_box_x_min > (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150/1.25) and anchor_distances_avg > 70:                
            hand_position_eval = "TOO_FAR"  
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)                           
        else:
            hand_position_eval = "TOO_CLOSE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)         

        x1 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - 150
        y1 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - 150
        x2 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + 150
        y2 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + 150      

        if x1 >= 0 and x2 <= frame_x_max and y1 >= 0 and y2 <= frame_y_max:
            test_frame_x1 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - 150
            test_frame_y1 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - 150
            test_frame_x2 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + 150
            test_frame_y2 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + 150                        

            forward_frame = test_frame[test_frame_y1:test_frame_y2, test_frame_x1:test_frame_x2]    
            forward_frame_blur_1 = cv2.blur(forward_frame,(5,5))
            forward_frame_blur_2 = cv2.medianBlur(forward_frame_blur_1,5)
            forward_frame_blur_3 = cv2.GaussianBlur(forward_frame_blur_2,(5,5),0)
            forward_frame_blur_4 = cv2.bilateralFilter(forward_frame_blur_3,9,75,75)                        

            forward_frame_hsv = cv2.cvtColor(forward_frame_blur_4, cv2.COLOR_BGR2HSV)
            forward_frame_mask = cv2.inRange(forward_frame_hsv,(135, 100, 20), (160, 255, 255))
            foward_threshold = cv2.threshold(forward_frame_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
            forward_inverse_threshold = cv2.threshold(foward_threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
            forward_inverse_threshold = cv2.flip(forward_inverse_threshold, 1)

            cv2.imshow('Hand Segmentation', forward_inverse_threshold)

            # this works ^ now need analytics frame to show
    else:
        print('reset')
    # video text here 
    
    if cv2.waitKey(25) == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
