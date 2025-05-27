import sys
import os
import cv2
import time
import threading
import numpy as np

# Path configuration
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # 10_team_code
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

YOLOV5_PATH = os.path.join(ROOT_DIR, "yolov5")  # 10_team_code/yolov5
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Import functions from voice_video_recognition which is in the YOLOV5_PATH
# Make sure the file exists in the yolov5 directory
sys.path.insert(0, YOLOV5_PATH)  # Prioritize this path for imports
from voice_video_recognition import (
    load_yolo_model,
    detect_objects,
    object_to_class_index
)

# Import hardware components
from Server.camera import Camera
from Server.motor import tankMotor
from Server.ultrasonic import Ultrasonic
from Server.servo import Servo

'''
PATROL_SPEED     = 500  
SIDE_MOVE_TIME   = 8.0  
SIDE_STOP_TIME   = 2.0  
TURN_MOVE_TIME   = 6.0  
TURN_STOP_TIME   = 0.5  

TURN_SPEED     = 700 

CENTER_SPEED     = 200  
CENTER_THRESH    = 0.05 
TURN_BASE_SPEED  = 800  
TURN_MIN_SPEED   = 200  
TURN_MAX_SPEED   = 1200 

APPROACH_SPEED   = 200  
STOP_DISTANCE_CM = 10
'''

SIDE_TOTAL_TIME   = 3.0    
INTERVAL_MOVE     = 0.5 
INTERVAL_DETECT   = 2.0    

TURN_TOTAL_TIME   = 1.0
TURN_INTERVAL_MOVE   = 0.3
TURN_INTERVAL_DETECT = 2.0

PATROL_SPEED      = 800   
TURN_SPEED        = 1100   

APPROACH_SPEED    = 200
STOP_DISTANCE_CM  = 10


def center_object_after_detection():
    """
    After detecting an object, control the motors to center the object in the frame
    """
    # Load model and setup devices
    weights_path = os.path.join(ROOT_DIR, 'model', 'converted_best.pt')
    model, device, imgsz = load_yolo_model(weights_path)
    model.eval()
    
    # Initialize hardware components
    cam = Camera()
    cam.start_stream()
    motor = tankMotor()
    ultrasonic = Ultrasonic()
    
    # Define target object
    target_object = "redBall"
    target_class_index = object_to_class_index.get(target_object)
    
    # Define control parameters
    frame_center_threshold = 0.05  # 5% of image width
    turn_base_speed = 1000         # Base turning speed
    max_turn_speed = 1500          # Maximum turning speed
    min_turn_speed = 800           # Minimum turning speed
    
    servo      = Servo()
    time.sleep(10.0)
    #servo.setServoAngle(0, 90)  
    #servo.setServoAngle(1, 90)
    
    servo_done = {"flag": False}

    def servo_task():
        print("[GRIP] Open claw...")
        servo.setServoAngle(0,  60)   
        time.sleep(1)

        print("[ARM] Lowering arm...")
        servo.setServoAngle(1,  60)    
        time.sleep(1)

        print("[GRIP] Closing claw...")
        servo.setServoAngle(0, 150)    
        time.sleep(1)

        print("[ARM] Raising arm with ball...")
        servo.setServoAngle(1, 150)   
        time.sleep(1)

        servo_done["flag"] = True
    
    try:
        print(f"[INFO] Looking for {target_class_index}...")
        
        '''
        print("[INFO]")
        found = False
        for side in range(4):
            motor.setMotorModel(PATROL_SPEED, PATROL_SPEED)
            t0 = time.time()
            print("patrol well detect")
            while time.time() - t0 < SIDE_MOVE_TIME:
                dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                if dets:
                    found = True
                    break
                cv2.imshow("Patrol", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    motor.setMotorModel(0,0)
                    return
                time.sleep(0.05)
            motor.setMotorModel(0, 0)
            if found:
                break

            motor.setMotorModel(TURN_SPEED, -TURN_SPEED)
            t0 = time.time()
            print("turn well detect")
            while time.time() - t0 < TURN_MOVE_TIME:
                dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                if dets:
                    found = True
                    break
                cv2.imshow("Patrol", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    motor.setMotorModel(0,0)
                    return
                time.sleep(0.05)
            motor.setMotorModel(0, 0)
            if found:
                break
 
        if not found:
            print("not found")
            while True:
                motor.setMotorModel(PATROL_SPEED, PATROL_SPEED)
                dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                if dets:
                    motor.setMotorModel(0,0)
                    found = True
                    break
                cv2.imshow("Search", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    motor.setMotorModel(0,0)
                    return
                time.sleep(0.1)
        '''
        path_log = []

        def record_and_drive(l_speed, r_speed, duration):
            motor.setMotorModel(l_speed, r_speed)
            path_log.append((l_speed, r_speed, duration))
            time.sleep(duration)
            motor.setMotorModel(0, 0)
         
        def record_and_drive_for_detect(l_speed, r_speed, duration):
            motor.setMotorModel(l_speed, r_speed)
            path_log.append((l_speed, r_speed, duration))
            time.sleep(duration)
                
        steps_per_side = int(SIDE_TOTAL_TIME / INTERVAL_MOVE)
        steps_per_turn = int(TURN_TOTAL_TIME / TURN_INTERVAL_MOVE)
        
        found = False
        print("[INFO] start patroling")
        for side in range(4):
            print(f"[PATROL] side {side+1} move {SIDE_TOTAL_TIME}s in {steps_per_side} steps")
            for _ in range(steps_per_side):
                print("move")
                record_and_drive(PATROL_SPEED, PATROL_SPEED, INTERVAL_MOVE)
                '''
                motor.setMotorModel(PATROL_SPEED, PATROL_SPEED)
                time.sleep(INTERVAL_MOVE)
                motor.setMotorModel(0,0)
                '''

                t0 = time.time()
                while time.time() - t0 < INTERVAL_DETECT:
                    dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                    if dets:
                        found = True
                        break
                    cv2.imshow("Patrol", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        motor.setMotorModel(0,0); return
                    time.sleep(0.05)
                if found:
                    break
                 
                for _ in range(steps_per_turn):
                    record_and_drive(TURN_SPEED, -TURN_SPEED, TURN_INTERVAL_MOVE)
                    '''
                    motor.setMotorModel(TURN_SPEED, -TURN_SPEED)
                    time.sleep(TURN_INTERVAL_MOVE)
                    motor.setMotorModel(0,0)
                    '''
                    t0 = time.time()
                    while time.time() - t0 < TURN_INTERVAL_DETECT:
                        dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                        if dets:
                            found = True
                            break
                        cv2.imshow("Patrol", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            motor.setMotorModel(0,0); return
                        time.sleep(0.05)
                    if found:
                        break
                if found:
                    break
                
                for _ in range(steps_per_turn):
                    record_and_drive(-TURN_SPEED, TURN_SPEED, TURN_INTERVAL_MOVE)
                    '''
                    motor.setMotorModel(TURN_SPEED, -TURN_SPEED)
                    time.sleep(TURN_INTERVAL_MOVE)
                    motor.setMotorModel(0,0)
                    '''
                    t0 = time.time()
                    while time.time() - t0 < TURN_INTERVAL_DETECT:
                        dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                        if dets:
                            found = True
                            break
                        cv2.imshow("Patrol", frame)
                        if cv2.waitKey(1) & 0xFF == 27:
                            motor.setMotorModel(0,0); return
                        time.sleep(0.05)
                    if found:
                        break
                
            motor.setMotorModel(0,0)
            if found:
                break

            for _ in range(steps_per_turn):
                record_and_drive(TURN_SPEED, -TURN_SPEED, TURN_INTERVAL_MOVE)
                '''
                motor.setMotorModel(TURN_SPEED, -TURN_SPEED)
                time.sleep(TURN_INTERVAL_MOVE)
                motor.setMotorModel(0,0)
                '''

                t0 = time.time()
                while time.time() - t0 < TURN_INTERVAL_DETECT:
                    dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                    if dets:
                        found = True
                        break
                    cv2.imshow("Patrol", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        motor.setMotorModel(0,0); return
                    time.sleep(0.05)
                if found:
                    break
            motor.setMotorModel(0,0)
            if found:
                break

        '''
        if not found:
            while True:
                motor.setMotorModel(PATROL_SPEED, PATROL_SPEED)
                dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                if dets:
                    motor.setMotorModel(0,0)
                    found = True
                    break
                cv2.imshow("Search", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    motor.setMotorModel(0,0); return
                time.sleep(0.1)
        '''
        if not found:
            while True:
                record_and_drive(PATROL_SPEED, PATROL_SPEED, 0.1)
                # motor.setMotorModel(PATROL_SPEED, PATROL_SPEED)
                print("test 1")
                dets, frame = detect_objects(cam, model, device, imgsz, [target_class_index] if target_class_index is not None else None)
                if dets:
                    motor.setMotorModel(0,0)
                    found = True
                    break
                cv2.imshow("Search", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    motor.setMotorModel(0,0); return
                # time.sleep(0.1)
      
        # Main detection and centering loop
        while True:
            # Check distance - stop if too close to an object
            dist = ultrasonic.get_distance()
            print(f"[INFO] Distance: {dist:.1f} cm")
            
            if dist <7:
                record_and_drive_for_detect(-200, -200, 0.8)
                '''
                motor.setMotorModel(-200, -200)
                time.sleep(0.8)
                '''
            
            if dist != 0 and dist < 10 and dist>7:  
                motor.setMotorModel(0, 0)
                print(f"[INFO] Arrived at the ball (distance {dist:.1f} cm).")
                           
                threading.Thread(target=servo_task, daemon=True).start()
                
                while not servo_done["flag"]:
                    time.sleep(0.05)

                servo.setServoStop()
                print("[INFO] Servo PWM stopped cleanly.")
                
                '''
                print("[GRIP] Open claw...")
                servo.setServoAngle('0', 90)    
                time.sleep(2)   

                print("[ARM] Lowering arm...")
                servo.setServoAngle('1', 90)    
                time.sleep(2)                 

                print("[GRIP] Closing claw...")
                servo.setServoAngle('0', 150)    
                time.sleep(2)
                
                print("[ARM] Raising arm with ball...")
                servo.setServoAngle('1', 150)   
                time.sleep(2)
                '''
                
                break
            
            # Get current detection results
            detections, frame = detect_objects(
                cam,
                model,
                device,
                imgsz,
                [target_class_index] if target_class_index is not None else None
            )
            
            # If no detection, continue searching
            if not detections:
                print("[WARN] No detection. Continuing search...")
                record_and_drive(800, 800, 0.1)
                '''
                motor.setMotorModel(200, 200)  # Move forward slowly while searching
                time.sleep(0.05)
                '''
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break
                continue

            # Find the detection with highest confidence
            best_detection = max(detections, key=lambda x: x['confidence'])
            
            # Calculate center X-coordinate of the object
            x1, y1 = map(int, best_detection["bbox"][0])
            x2, y2 = map(int, best_detection["bbox"][2])
            object_center_x = (x1 + x2) // 2
            
            # Frame dimensions and center calculation
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2
            
            # Calculate offset between object center and frame center
            offset = object_center_x - frame_center_x
            
            # Calculate threshold based on frame width
            center_threshold = int(frame_width * frame_center_threshold)
            
            # Display information on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame, (frame_center_x, 0), (frame_center_x, frame_height), (0, 255, 0), 1)
            cv2.line(frame, (object_center_x, 0), (object_center_x, frame_height), (255, 0, 0), 1)
            cv2.putText(frame, f'Offset: {offset}px', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Confidence: {best_detection["confidence"]:.2f}', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f'Distance: {dist:.1f} cm', (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
            
            # Check if centered
            if abs(offset) < center_threshold:
                print(f"[SUCCESS] Object centered! Offset: {offset}px, Moving forward")
                # Object is centered, move forward
                # record_and_drive_for_detect(200, 200, 0.4)
                record_and_drive_for_detect(200, 200, 0.4)
                '''
                motor.setMotorModel(200, 200)
                time.sleep(0.4)
                '''
            else:
                # Calculate turn speed based on offset
                turn_speed = min(max(abs(offset) / (frame_width / 4) * turn_base_speed, min_turn_speed), max_turn_speed)
                
                if offset > 0:  # Object on right, need to turn right
                    # Right turn: left wheel faster, right wheel slower
                    left_speed = turn_speed * 0.4
                    right_speed = turn_speed 
                    print(f"[ADJUST] Turning RIGHT. Offset: {offset}px")
                else:  # Object on left, need to turn left
                    # Left turn: left wheel slower, right wheel faster
                    left_speed = turn_speed 
                    right_speed = turn_speed* 0.4
                    print(f"[ADJUST] Turning LEFT. Offset: {offset}px")
                
                # Execute the turn while moving forward
                # record_and_drive_for_detect(int(left_speed), int(right_speed), 0.5)
                record_and_drive(int(left_speed), int(right_speed), 0.2)
                '''
                motor.setMotorModel(int(left_speed), int(right_speed))
                time.sleep(0.5)
                '''
                # record_and_drive_for_detect(800, 800, 0.1)
                record_and_drive(800, 800, 0.1)
                '''
                motor.setMotorModel(500,500)
                time.sleep(0.05)
                '''
            
            # Short sleep to avoid overwhelming the system
            time.sleep(0.5)
            
        print("return to original position")
    
        for l, r, dur in reversed(path_log):
            motor.setMotorModel(-l, -r)
            time.sleep(dur)
            motor.setMotorModel(0,0)
            time.sleep(0.2)
        
        for _ in range(steps_per_turn):
            record_and_drive(-TURN_SPEED, TURN_SPEED, TURN_INTERVAL_MOVE)
        
        for _ in range(2):
            record_and_drive(PATROL_SPEED, PATROL_SPEED, INTERVAL_MOVE)
            
        for _ in range(steps_per_turn):
            record_and_drive(-TURN_SPEED, TURN_SPEED, TURN_INTERVAL_MOVE)
            
        for _ in range(4):
            record_and_drive(PATROL_SPEED, PATROL_SPEED, INTERVAL_MOVE)
            
        motor.setMotorModel(0,0)
  
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    
    except Exception as e:
        print(f"[ERROR] An exception occurred: {e}")
    
    finally:
        # Ensure motors are stopped
        motor.setMotorModel(0, 0)
        motor.close()
        
        # Clean up camera resources
        print("[INFO] Cleaning up camera resources...")
        cam.stop_stream()
        cam.close()
        print("[INFO] Done.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("[INFO] Launching object centering system...")
    # Print the current path and YOLOV5_PATH for debugging
    print(f"Current directory: {os.getcwd()}")
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"YOLOV5_PATH: {YOLOV5_PATH}")
    print(f"Python path: {sys.path}")
    
    # Check if the file exists
    voice_video_file = os.path.join(YOLOV5_PATH, "voice_video_recognition.py")
    if os.path.exists(voice_video_file):
        print(f"Found voice_video_recognition.py at {voice_video_file}")
    else:
        print(f"ERROR: Could not find voice_video_recognition.py at {voice_video_file}")
        # Try to find it elsewhere
        for root, dirs, files in os.walk(ROOT_DIR):
            if "voice_video_recognition.py" in files:
                print(f"Found voice_video_recognition.py at {os.path.join(root, 'voice_video_recognition.py')}")
    
    # Run the main function
    center_object_after_detection()
    print("[INFO] Object centering completed.")
