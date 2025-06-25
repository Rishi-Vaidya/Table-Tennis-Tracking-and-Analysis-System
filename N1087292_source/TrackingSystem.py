import cv2
import numpy as np
import sys
import time

#load the video
video_path = "c:/Users/rishi/OneDrive - Nottingham Trent University/YEAR 3/Final Year Project/N1087292_source/videos/gameplay1.mp4"
print(f"Trying to open video: {video_path}")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

#general video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {frame_width}x{frame_height}, FPS: {fps}")


playback_speed = 0.2  #sets video slomo 20% speed
delay = int(1000 / (fps * playback_speed))

#window size taken up
window_width = frame_width // 2
window_height = frame_height // 2

#create dispsplay winndows
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original', window_width, window_height)

cv2.namedWindow('Ball Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Ball Detection', window_width, window_height)

cv2.namedWindow('Controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Controls', 400, 350)
cv2.moveWindow('Controls', window_width + 50, 50)

#trackbars to help tune paramters for tracking/optimisation essentially
def nothing(x):
    pass

#trackbars for tuning features, defaults set to ideal discovered
cv2.createTrackbar('Brightness Threshold', 'Controls', 200, 255, nothing)
cv2.createTrackbar('Min Ball Size', 'Controls', 8, 50, nothing)
cv2.createTrackbar('Max Ball Size', 'Controls', 10, 50, nothing)
cv2.createTrackbar('Min Circularity (%)', 'Controls', 75, 100, nothing)
cv2.createTrackbar('Blur', 'Controls', 5, 15, nothing)
cv2.createTrackbar('Brightness Boost', 'Controls', 14, 100, nothing)
cv2.createTrackbar('Use Background Sub', 'Controls', 1, 1, nothing)  
cv2.createTrackbar('BG Learning Rate (%)', 'Controls', 10, 100, nothing)  
cv2.createTrackbar('Motion Threshold', 'Controls', 18, 50, nothing)  
cv2.createTrackbar('Min Motion Area', 'Controls', 3, 50, nothing)  #Min size for motion detection
cv2.createTrackbar('Trail Length (sec)', 'Controls', 2, 5, nothing)  #Trail length in seconds

#boundaries defined for the table area so where the ball will be detected 
TABLE_Y_MIN = int(frame_height * 0.65) 
TABLE_Y_MAX = int(frame_height * 0.75)  
TABLE_X_MIN = int(frame_width * 0.36)  
TABLE_X_MAX = int(frame_width * 0.78)  

#initialize tracking variables
ball_positions = []  #detected ball positions with timestamp
table_ball_positions = []  #ball position specifically within table boundaries
bounces = []
paused = False
show_mask = True
frame_count = 0
is_tracking = False  #see if detection is happening
track_threshold = 30  #max distance to ensure ball is definitely tracked


current_time = 0  #current time in video 
frame_time = 1.0 / fps  #time per frame in seconds

#background subractor to help with detection 
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=20, detectShadows=False)

#helps with motion detection
prev_gray = None

#initialise for frame skipping 
frame_buffer = []
frame_skip = 2  

#read the first frame to initialize
ret, first_frame = cap.read()
if not ret:
    print("Failed to read the first frame.")
    exit()

#reset to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#for predicting next ball position
def predict_next_position(positions, max_frames=5):
    if len(positions) < 2:
        return None
    
    #use only recent positions
    recent = [pos for pos, _ in positions[-min(max_frames, len(positions)):]]
    
    #calculate average velocity
    if len(recent) >= 2:
        x_vel = sum([recent[i][0] - recent[i-1][0] for i in range(1, len(recent))]) / (len(recent) - 1)
        y_vel = sum([recent[i][1] - recent[i-1][1] for i in range(1, len(recent))]) / (len(recent) - 1)
        
        #predict next position
        next_x = int(recent[-1][0] + x_vel)
        next_y = int(recent[-1][1] + y_vel)
        
        return (next_x, next_y)
    
    return None

while True:
    #frame handling with pause capability
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        current_frame = frame.copy()
        
        #store frames for motion detection
        frame_buffer.append(frame.copy())
        if len(frame_buffer) > frame_skip + 1:
            frame_buffer.pop(0)
        
        #update current time
        current_time += frame_time
    else:
        frame = current_frame.copy()
    
    #resize for display
    display_frame = cv2.resize(frame, (window_width, window_height))
    
    #get current settings
    brightness_threshold = cv2.getTrackbarPos('Brightness Threshold', 'Controls')
    min_ball_size = cv2.getTrackbarPos('Min Ball Size', 'Controls')
    max_ball_size = cv2.getTrackbarPos('Max Ball Size', 'Controls')
    min_circularity = cv2.getTrackbarPos('Min Circularity (%)', 'Controls') / 100.0
    blur_size = cv2.getTrackbarPos('Blur', 'Controls')
    if blur_size % 2 == 0:  #ensure blur size is odd
        blur_size += 1
    brightness_boost = cv2.getTrackbarPos('Brightness Boost', 'Controls')
    
    use_bg_subtraction = cv2.getTrackbarPos('Use Background Sub', 'Controls') == 1
    bg_learning_rate = cv2.getTrackbarPos('BG Learning Rate (%)', 'Controls') / 1000.0  # Slow learning rate
    motion_threshold = cv2.getTrackbarPos('Motion Threshold', 'Controls')
    min_motion_area = cv2.getTrackbarPos('Min Motion Area', 'Controls')
    trail_length = cv2.getTrackbarPos('Trail Length (sec)', 'Controls')
    
    #Create a working copy of the frame at original resolution for processing
    working_frame = frame.copy()
    
   
    #Convert to grayscale
    gray = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
    
    #boost brightness to help detect white ball
    if brightness_boost > 0:
        gray = cv2.convertScaleAbs(gray, alpha=1.0, beta=brightness_boost)
    
    #apply blur to reduce noise
    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    #initialize the mask for ball candidates
    ball_mask = np.zeros_like(gray)

    
    #basic thresholding for bright objects
    _, thresh_basic = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    
    #background subtraction to remove static bright areas
    if use_bg_subtraction:
        fg_mask = bg_subtractor.apply(working_frame, learningRate=bg_learning_rate)
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    else:
        fg_thresh = np.zeros_like(gray)
    
    #motion detection by frame differencing
    motion_mask = np.zeros_like(gray)
    if len(frame_buffer) > frame_skip:
        previous = cv2.cvtColor(frame_buffer[0], cv2.COLOR_BGR2GRAY)
        previous = cv2.GaussianBlur(previous, (blur_size, blur_size), 0)
        
        #calculate absolute difference between current and previous mask
        frame_diff = cv2.absdiff(gray, previous)
        _, motion_mask = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)
        
        #clean up motion mask
        kernel = np.ones((3, 3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
    
    combined_mask = cv2.bitwise_and(thresh_basic, cv2.bitwise_or(fg_thresh, motion_mask))
    
    #apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    #create a color mask for visualization
    mask_display = np.zeros((window_height, window_width, 3), dtype=np.uint8)
    
    #show different detection methods side by side for debugging
    thresh_basic_resized = cv2.resize(thresh_basic, (window_width//2, window_height//2))
    fg_thresh_resized = cv2.resize(fg_thresh, (window_width//2, window_height//2))
    motion_mask_resized = cv2.resize(motion_mask, (window_width//2, window_height//2))
    combined_mask_resized = cv2.resize(combined_mask, (window_width//2, window_height//2))
    
    #place the masks in each quadrant
    mask_display[0:window_height//2, 0:window_width//2] = cv2.cvtColor(thresh_basic_resized, cv2.COLOR_GRAY2BGR)
    mask_display[0:window_height//2, window_width//2:window_width] = cv2.cvtColor(fg_thresh_resized, cv2.COLOR_GRAY2BGR)
    mask_display[window_height//2:window_height, 0:window_width//2] = cv2.cvtColor(motion_mask_resized, cv2.COLOR_GRAY2BGR)
    mask_display[window_height//2:window_height, window_width//2:window_width] = cv2.cvtColor(combined_mask_resized, cv2.COLOR_GRAY2BGR)
    
    #add labels
    cv2.putText(mask_display, "Brightness", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(mask_display, "Background", (window_width//2 + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(mask_display, "Motion", (10, window_height//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(mask_display, "Combined", (window_width//2 + 10, window_height//2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    #find contours on the combined mask 
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #prepare result image
    result = display_frame.copy()
    
    #draw table boundaries
    cv2.line(result, (0, int(TABLE_Y_MIN/2)), (window_width, int(TABLE_Y_MIN/2)), (0, 255, 255), 1)
    cv2.line(result, (0, int(TABLE_Y_MAX/2)), (window_width, int(TABLE_Y_MAX/2)), (0, 255, 255), 1)
    cv2.line(result, (int(TABLE_X_MIN/2), 0), (int(TABLE_X_MIN/2), window_height), (0, 255, 255), 1)
    cv2.line(result, (int(TABLE_X_MAX/2), 0), (int(TABLE_X_MAX/2), window_height), (0, 255, 255), 1)


    
    #process each contour to find the ball
    best_ball = None
    best_score = 0
    
    scale_factor = window_width / frame_width  #for scaling coordinates to display size
    
    #predict next position based on previous trajectory
    predicted_position = predict_next_position(ball_positions) if ball_positions else None
    
    for contour in contours:
        #calculate area
        area = cv2.contourArea(contour)
        
        #filter by size
        if area < min_ball_size or area > max_ball_size * max_ball_size * np.pi:
            continue
        
        #calculate circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        #filter by circularity
        if circularity < min_circularity:
            continue
        
        #calculate bounding circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        #scale to display size
        display_center = (int(x * scale_factor), int(y * scale_factor))
        display_radius = int(radius * scale_factor)
        
        #Score this candidate
        size_score = 1.0 - abs(radius - 10) / 10 if radius <= 20 else 0
        shape_score = circularity
        
        #Check if it's in the table region 
        in_table_region = TABLE_X_MIN <= x <= TABLE_X_MAX and TABLE_Y_MIN <= y <= TABLE_Y_MAX

        position_score = 1.0 if in_table_region else 0.5
        
        #add motion score with higher score if motion detected at location
        motion_score = 0
        if len(frame_buffer) > frame_skip:
            motion_val = motion_mask[int(y), int(x)]
            motion_score = motion_val / 255
        
        #Add trajectory continuity score - higher if close to predicted position
        trajectory_score = 0
        if predicted_position and is_tracking:
            pred_x, pred_y = int(predicted_position[0] / scale_factor), int(predicted_position[1] / scale_factor)
            distance = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
            #normalize distance (closer = higher score)
            trajectory_score = max(0, 1 - distance / track_threshold)
        
        
        # Give more weight to trajectory continuity when actively tracking
        if is_tracking and predicted_position:
            score = (size_score * 0.2 + shape_score * 0.2 + position_score * 0.1 + 
                    motion_score * 0.2 + trajectory_score * 0.3)
        else:
            #Default weights when not tracking or no prediction
            score = size_score * 0.3 + shape_score * 0.3 + position_score * 0.2 + motion_score * 0.2
        
        
        color = (0, int(255 * circularity), int(255 * (1-circularity)))
        cv2.circle(mask_display[window_height//2:window_height, window_width//2:window_width], 
                  (int(x * scale_factor - window_width//2), int(y * scale_factor - window_height//2)), 
                  display_radius, color, 1)
        
        #update best ball if this is better
        if score > best_score:
            best_score = score
            best_ball = (display_center, display_radius, score, in_table_region)
    
    #if a ball was found
    if best_ball is not None:
        center, radius, score, in_table_region = best_ball
        is_tracking = True  #tracking succesful 
        
        #save ball position with timestamp
        ball_positions.append((center, current_time))
        
        #remove positions older than the trail length
        cutoff_time = current_time - trail_length
        ball_positions = [pos for pos in ball_positions if pos[1] > cutoff_time]
        
        # Track positions within table region separately for bounce detection
        if in_table_region:
            table_ball_positions.append(center)

            if len(table_ball_positions) > 30:
                table_ball_positions.pop(0)
        
        #Draw ball on result
        cv2.circle(result, center, radius, (0, 255, 0), 2)
        cv2.circle(result, center, 2, (0, 0, 255), -1)
        
        #Draw confidence score
        cv2.putText(result, f"Score: {score:.2f}", (center[0] + 10, center[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
       
        #detect bounces using position history - only when within table region
        if len(table_ball_positions) > 5:
            
            #looking for changes in verticality of ball
            y_deltas = [table_ball_positions[i+1][1] - table_ball_positions[i][1] for i in range(len(table_ball_positions)-5, len(table_ball_positions)-1)]
            going_down = sum(y_deltas[:2]) > 0
            going_up = sum(y_deltas[2:]) < 0
            
            #check if the ball was going down and is now going up
            if going_down and going_up:
                #convert display coordinates back to original frame coordinates if needed later
                original_x = center[0] / scale_factor
                original_y = center[1] / scale_factor
                
                #check if the bounce is within the table boundaries
                if (TABLE_X_MIN <= original_x <= TABLE_X_MAX and 
                    TABLE_Y_MIN <= original_y <= TABLE_Y_MAX):
                    
                    # Make sure this is a new bounce
                    is_new_bounce = True
                    for bx, by in bounces:
                        distance = np.sqrt((center[0] - bx)**2 + (center[1] - by)**2)
                        if distance < 10:  #distance to previous bounce
                            is_new_bounce = False
                            break
                    
                    if is_new_bounce:
                        bounces.append(center)
                        print(f"Bounce detected at {center} ")


    else:
        #If no ball detected for several frames, stop tracking
        if is_tracking and frame_count % 10 == 0:  # Check every 10 frames
            is_tracking = False
    
    #draw temporary trajectory - only showing positions within the trail time window
    if len(ball_positions) > 1:
        #draw lines between consecutive points
        for i in range(1, len(ball_positions)):
            #get positions from the timestamp tuples
            pos1, time1 = ball_positions[i-1]
            pos2, time2 = ball_positions[i]
            
            
            age_factor = (current_time - time2) / trail_length 
        
            #Limit age factor to be between 0 and 1
            age_factor = max(0, min(1, age_factor))
        
        #Calculate alpha (opacity) based on age - newer points are more visible
        alpha = 1.0 - age_factor
        
        #Only draw if alpha is significant enough
        if alpha > 0.1:  #Skip very faint lines
            
            color = (int(255 * alpha), 0, 0, int(255 * (1 - alpha)))  #orange to red line for tracking
            
            #Draw the line with the faded color
            cv2.line(result, pos1, pos2, color, 1)
    
    #Draw bounce points
    for i, (bx, by) in enumerate(bounces):
        cv2.circle(result, (bx, by), 3, (255, 255, 255), -1)
        cv2.putText(result, f"#{i+1}", (bx + 10, by), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
     #Draw predicted next position
    if predicted_position and is_tracking:
        cv2.circle(result, predicted_position, 4, (255, 255, 0), -1)  #yellow dot for prediction
    
    #Show frame count and other info
    frame_count += 1
    cv2.putText(result, f"Frame: {frame_count}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Time: {current_time:.2f}s", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Bounces: {len(bounces)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Ball detected: {'Yes' if best_ball else 'No'}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(result, f"Trail length: {trail_length}s", (10, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    #Display status on Controls window
    controls_display = np.zeros((300, 350, 3), dtype=np.uint8)
    cv2.putText(controls_display, "CONTROLS:", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_display, "P: Pause/Play", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, "S: Step frame (when paused)", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, "M: Toggle mask view", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, "C: Clear bounces", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, "T: Clear trajectory", (10, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, "Q: Quit", (10, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(controls_display, f"STATUS:", (10, 190), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(controls_display, f"{'PAUSED' if paused else 'PLAYING'}", (10, 210), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if not paused else (0, 0, 255), 1)
    cv2.putText(controls_display, f"Detected objects: {len(contours)}", (10, 230), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, f"Ball confidence: {best_score:.2f}" if best_ball else "No ball detected", (10, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(controls_display, f"Trail points: {len(ball_positions)}", (10, 270), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    #Show the frames
    cv2.imshow('Original', result)
    cv2.imshow('Ball Detection', mask_display)
    cv2.imshow('Controls', controls_display)
    
    # key operations 
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):  # ause/Resume
        paused = not paused
    elif key == ord('s') and paused:  #Step forward when paused
        ret, current_frame = cap.read()
        if not ret:
            print("End of video.")
            break
        #Add to buffer for motion detection
        frame_buffer.append(current_frame.copy())
        if len(frame_buffer) > frame_skip + 1:
            frame_buffer.pop(0)
        #Update time when stepping
        current_time += frame_time
    elif key == ord('m'):  #Toggle mask view
        show_mask = not show_mask
    elif key == ord('c'):  #Clear bounces
        bounces = []
        print("Bounces cleared")
    elif key == ord('t'):  #Clear trajectory
        ball_positions = []
        table_ball_positions = []
        is_tracking = False
        print("Trajectory cleared")
    elif key == ord('+') or key == ord('='):  # +/= for slowing vid down
        playback_speed = max(0.1, playback_speed - 0.1)
        delay = int(1000 / (fps * playback_speed))
        print(f"Playback speed: {playback_speed:.1f}x")
    elif key == ord('-'):  #- for speeding vid up
        playback_speed += 0.1
        delay = int(1000 / (fps * playback_speed))
        print(f"Playback speed: {playback_speed:.1f}x")

#Save bounce points to file before exiting
if bounces:
    with open(r"C:\Users\rishi\OneDrive - Nottingham Trent University\YEAR 3\Final Year Project\N1087292_source\bounce_points.txt", "w") as f:
        f.write("Bounce Points (x, y):\n")
        for i, (x, y) in enumerate(bounces):
            f.write(f"Point {i+1}: ({x}, {y})\n")
    print(f"Saved {len(bounces)} bounce points to bounce_points.txt")


cap.release()
cv2.destroyAllWindows()