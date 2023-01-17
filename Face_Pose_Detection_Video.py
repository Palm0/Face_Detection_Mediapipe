# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:08:18 2023

@author: Chris
"""
import os
import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import matplotlib as plt
from pylab import *

# Initalisierung der Benötigten Variablen aus Mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        

def rescale_frame(frame, percent):
   
    """
    Herrausziehen der einzelnen Frames und bestimmung wie viel Prozent der Gesamtanzahl der Frames Verwendet werden soll.
    https://www.kaggle.com/code/brodielamont/test-opencv-video-and-mediapipe
    """
    
    width = int(frame.shape[1] * percent/100)
    height = int(frame.shape[0] * percent/100)
    dimention = (width, height)
    
    return cv2.resize(frame, dimention, interpolation = cv2.INTER_AREA)    


def get_Data_pose(path, model,n,n_plot):
    
    """
    Funktioniert. Gibt aber bei manchen Frames einen Attribut error zurück, weil es in diesen Frame die Pose nicht auslesen kann.
    Woran das liegt bin ich mir noch nicht sicher, weil der Kamerawinkel eigentlich garkeine Rolle spielen sollte. Dadurch kann er an 
    manchen Punkten keine x,y,z Koordinate auslesen weswegen es zu einen Error kommt.
    https://google.github.io/mediapipe/solutions/pose.html
    https://mlhive.com/2021/11/person-pose-landmarks-detection-using-mediapipe
    https://www.kaggle.com/code/brodielamont/test-opencv-video-and-mediapipe
    """
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, min_tracking_confidence=0.5,enable_segmentation=False, model_complexity=model) as pose:
    
        # Import video file
        video_cap = cv2.VideoCapture(DATA_DIRECTORY+"/"+path)

        if (video_cap.isOpened()== False): 
            print("Error opening video  file")

        success, image = video_cap.read()

        if success:
            print("success")

        # Resize video
        # Extract frames
        count = 1
        xyz = []
        error_l = []
        while success:
            try:
                image = rescale_frame(image, n)
                result = pose.process(image)
                x = result.pose_landmarks.landmark[0].x
                y = result.pose_landmarks.landmark[0].y
                z = result.pose_landmarks.landmark[0].z
                xyz.append([x,y,z])
                
                cv2.imwrite(FRAME_DIRECTORY+"/"+path[:-4]+"_"+str(count)+".jpeg", image)
                success, image = video_cap.read()
                count = count + 1
                
                if success==False:
                    break
                
                
                plot_landmark_pose(image,result,n_plot,count)
                
            except AttributeError:
                error_l.append(f"Attribute Error für Posedetection bei Frame: {count}")
                print(f"Attribute Error für Posedetection bei Frame: {count}")
                cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path[:-4]+"_Frame_"+str(count)+".jpeg", image)
                count = count + 1
        
        data = np.array(xyz)
        saving_csv(path[:-4],data)
        saving_error(path[:-4], error_l)
        
        return data

def get_Data_mesh(path, model,n,n_plot):
    
    """
    Erkennt das Gesicht auf dem Video sehr gut wenn es einigermaßen zentral in dem Video ist. Sobald man das Gesicht von der Seite sieht
    oder nur zur Hälfte bekommt der Algorithmus Keine Informationen zurück und schmeißt einen Error. Beispiel: Video 001 wird alles sehr 
    gut erkannt - in Video 016 hingegen gibt der Algorithmus nichts zurück, da er das Gesicht nicht erkennt.
    
    Was man versuchen kann ist bevor man den Facemesh durchführt. Eine Face detection davor zu bauen um darüber dann den Facemesh 
    berechnen zu können. Wird gerade ausprobiert.
    https://github.com/serengil/tensorflow-101/blob/master/python/Mediapipe-Face-Detector.ipynb
    https://www.kaggle.com/code/brodielamont/test-opencv-video-and-mediapipe
    """
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, min_detection_confidence=0.5, max_num_faces=model) as face_mesh:
    
        # Import video file
        video_cap = cv2.VideoCapture(DATA_DIRECTORY+"/"+path)

        if (video_cap.isOpened()== False): 
            print("Error opening video  file")

        success, image = video_cap.read()
        
        print()
        if success:
            print("success")

        # Resize video
        # Extract frames
        count = 1
        xyz = []
        error_l = []
        while success:
            try:
                image = rescale_frame(image, n)
                result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                x = result.multi_face_landmarks[0].landmark[0].x
                y = result.multi_face_landmarks[0].landmark[0].y
                z = result.multi_face_landmarks[0].landmark[0].z
                xyz.append([x,y,z])
                
                cv2.imwrite(FRAME_DIRECTORY+"/"+path[:-4]+"_"+str(count)+".jpeg", image)
                success, image = video_cap.read()
              
                
                if success==False:
                    break
                
               
                count = count + 1
                
                
                landmarks = result.multi_face_landmarks[0]
                plot_landmark(image,landmarks,n_plot,count)
            
            except TypeError:
                error_l.append(f"Type Error für Facedetection bei Frame: {count}")
                print(f"Type Error für Facedetection bei Frame: {count}")
                cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path[:-4]+"_Frame_"+str(count)+".jpeg", image)
                count = count + 1
        
        data = np.array(xyz)
        saving_csv(path[:-4],data)
        saving_error(path[:-4], error_l)
        
        return data

def get_Data_detection(path, model,n,n_plot):
    
    """
    Face Detection Funktioniert bekommt aber manchmal einen TypeError, das liegt daran das es in Manchen Videos Schwierigkeiten hat das 
    Gesicht zu erkennen. Am Besten funktioniert der Algorithmus wenn man das Gesicht frontal aufgenommen hat in dem Video sobald ein 
    etwas zu Kleiner Gesichtsteil nur von der Seite zu sehen ist, bekommt man einen Nonetype zurück was dann den Fehler des TypeErrors 
    auslöst.
    So mit ist die Bedingung das das Gesicht gut erkennbar ist.
    Aber er funktioniert auch ganz solide wenn man eine Schräge Kameraperspektive hat. Sie darf nur einen gewissen winkel nicht 
    überschreiten bzw. Das Gesicht muss gut erkennbar sein.
    https://pylessons.com/face-detection
    https://www.kaggle.com/code/brodielamont/test-opencv-video-and-mediapipe
    """
    
    with mp_face_detection.FaceDetection(model_selection=model, min_detection_confidence=0.5) as face_detection:
    
        # Import video file
        video_cap = cv2.VideoCapture(DATA_DIRECTORY+"/"+path)

        if (video_cap.isOpened()== False): 
            print("Error opening video  file")

        success, image = video_cap.read()
        
        print()
        if success:
            print("success")

        # Resize video
        # Extract frames
        count = 1
        xyz = []
        while success:
            image = rescale_frame(image, n)
            result = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            result_face_d = result.detections
            
            cv2.imwrite("image"+str(count)+".jpeg", image)
            success, image = video_cap.read()
            
          
            if success==False:
                break
            
            try:
                xyz = tlbr(image,result_face_d)
            except TypeError:
                print("TypeError")
                
            count = count + 1
            
            if count % n_plot == 0:
                plot_Face_Detect(image,result_face_d)
        
        data = np.array(xyz)
        
        return data
    

def get_Data_holistic(path, model,n,n_plot):
    
    """
    
    https://github.com/google/mediapipe/blob/master/mediapipe/python/solutions/holistic.py
    
    """
    
    with mp_holistic.Holistic(static_image_mode=True, refine_face_landmarks=True, model_complexity=model,enable_segmentation=True) as holistic:
    
        # Import video file
        video_cap = cv2.VideoCapture(DATA_DIRECTORY+"/"+path)
        
        fps = video_cap.get(cv2.CAP_PROP_FPS)
        #width  = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
        #height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
        
        if (video_cap.isOpened()== False): 
            print("Error opening video  file")

        success, image = video_cap.read()
        
        print()
        if success:
            print("success")

        # Resize video
        # Extract frames
        count = 1
        xyz_pose = []
        xyz_face = []
        xyz_hand_r = []
        xyz_hand_l = []
        vis_l = []
        error_l = []
        
        while success:
            
                image = rescale_frame(image, n)
                result = holistic.process(image)
                
                
                # outputs=['pose_landmarks', 'pose_world_landmarks', 'left_hand_landmarks','right_hand_landmarks', 
                # 'face_landmarks', 'segmentation_mask']
                # Kann alles aus Results ausgelesen werden !!
                #print(result.pose_landmarks.landmarks[0])
                
                try:
                    x = result.pose_landmarks.landmark[0].x
                    y = result.pose_landmarks.landmark[0].y
                    z = result.pose_landmarks.landmark[0].z
                    
                    xyz_pose.append([x,y,z])
                
                
                except AttributeError:
                    error_l.append(f"Attribute Error für Posedetection bei Frame: {count}")
                    print(f"Attribute Error für Posedetection bei Frame: {count}")
                    cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path[:-4]+"_Frame_"+str(count)+".jpeg", image)
                    count = count + 1
                
                
                try:
                   
                    x = result.face_landmarks.landmark[0].x
                    y = result.face_landmarks.landmark[0].y
                    z = result.face_landmarks.landmark[0].z
                    
                    xyz_face.append([x,y,z])
                except:
                    error_l.append(f"Attribute Error für Facedetection bei Frame: {count}")
                    print(f"Attribute Error für Facedetection bei Frame: {count}")
                    cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path[:-4]+"_Frame_"+str(count)+".jpeg", image)
                    count = count + 1
                
                """
                try:
                    x = result.right_hand_landmarks.landmark[0].x
                    y = result.right_hand_landmarks.landmark[0].y
                    z = result.right_hand_landmarks.landmark[0].z
                    
                    xyz_hand_r.append([x,y,z])
                    
                    x = result.left_hand_landmarks.landmark[0].x
                    y = result.left_hand_landmarks.landmark[0].y
                    z = result.left_hand_landmarks.landmark[0].z
                    
                    xyz_hand_l.append([x,y,z])
                except AttributeError:
                     print(f"Attribute Error für Handdetection bei Frame: {count}")
                     count = count + 1
                    
                """
                try:
                    cv2.imwrite(FRAME_DIRECTORY+"/"+path[:-4]+"_"+str(count)+".jpeg", image)
                    success, image = video_cap.read()
                    
                   
                    if success==False:
                        break
                
                    # Nicht jeder Frame würd durch die IF-Abfrage 
                    # gezeichnet sondern nur jeder zehnte
                    
                    #if count % 10 == 0:
                    helper = count
                    landmarks = result.face_landmarks
                    plotting_landmarks_f_p(image,landmarks,result,path[:-4],n_plot,helper)
                        
                    count = count + 1
               
                except AttributeError:
                   error_l.append(f"Attribute Error für Plotting bei Frame: {count}")
                   print(f"Attribute Error für Plotting bei Frame: {count}")
                   cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path[:-4]+"_Frame_"+str(count)+".jpeg", image)
                   count = count + 1
                   
        vis_data = np.array(vis_l)
        data = np.array([xyz_pose,xyz_face])
        saving_error(path[:-4],error_l)
        saving_Video(path[:-4],count,fps)
        saving_csv(path[:-4],data)
        
        return data , vis_data

def tlbr(frame, mp_detections):
    
        """Return coorinates in typing.Iterable([[Top, Left, Bottom, Right]])

        Args:
            frame: (np.ndarray) - frame on which we want to apply detections
            mp_detections: (typing.List) - list of media pipe detections

        Returns:
            detections: (np.ndarray) - list of detection in [Top, Left, Bottom, Right] coordinates
        
        https://pylessons.com/face-detection
        """
    
        detections = []
        frame_height, frame_width, _ = frame.shape
        for detection in mp_detections:
            height = int(detection.location_data.relative_bounding_box.height * frame_height)
            width = int(detection.location_data.relative_bounding_box.width * frame_width)
            left = int(detection.location_data.relative_bounding_box.xmin * frame_width)
            top = int(detection.location_data.relative_bounding_box.ymin * frame_height)

            detections.append([top, left, top + height, left + width])

        return np.array(detections)

def plot_Face_Detect(frame,result_face_d, return_tlbr = False, mp_drawing_utils = True):
    
    """Main function to do face detection

        Args:
            frame: (np.ndarray) - frame to excecute face detection on
            return_tlbr: (bool) - bool option to return coordinates instead of frame with drawn detections

        Returns:
            typing.Union[
                frame: (np.ndarray) - processed frame with detected faces,
                detections: (typing.List) - detections in [Top, Left, Bottom, Right]
                ]
        
        https://pylessons.com/face-detection
        """
    
    if result_face_d:
        
            img = frame.copy()
            color = (255, 255, 255)
            thickness = 2
            
            if return_tlbr:
                return tlbr(frame, result_face_d.detections)

            if mp_drawing_utils:
                # Draw face detections of each face using media pipe drawing utils.
                for detection in result_face_d:
                    mp_drawing.draw_detection(img, detection)
                    img = cv2.resize(img, (384*2,288*2))
                    
                    
                    cv2.imshow('image',img)
                    cv2.waitKey(0)
    
            else:
                # Draw face detections of each face using our own tlbr and cv2.rectangle
                for tlbr_t in tlbr(frame, result_face_d):
                    cv2.rectangle(frame, tlbr_t[:2][::-1], tlbr_t[2:][::-1], color, thickness)
                    
    return frame


def plot_landmark(img_base,landmarks,n_plot,count):
    
    img = img_base.copy()
    
    for idx, landmark in enumerate(landmarks.landmark):
        x = landmark.x
        y = landmark.y
    
        relative_x = int(img.shape[1] * x)
        relative_y = int(img.shape[0] * y)
    
        #print(relative_x, relative_y)
        cv2.circle(img, (relative_x, relative_y), 1, (0, 0, 255), -1)
    
    img = cv2.resize(img, (384*2,288*2))
    
    if count % n_plot == 0:
       cv2.imshow('image',img)
       cv2.waitKey(0)


def plot_landmark_pose(img_base,results,n_plot,count):
    
    annotated_image = img_base.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS)
    annotated_image = cv2.resize(annotated_image, (384*2,288*2))
    
    
    if count % n_plot == 0:
       cv2.imshow('image',annotated_image)
       cv2.waitKey(0)

def plotting_landmarks_f_p(img_base,landmarks,results,path,n_plot,count):
   
    img = img_base.copy()
    
    for idx, landmark in enumerate(landmarks.landmark):
        x = landmark.x
        y = landmark.y
    
        relative_x = int(img.shape[1] * x)
        relative_y = int(img.shape[0] * y)
    
        #print(relative_x, relative_y)
        cv2.circle(img, (relative_x, relative_y), 1, (0, 0, 255), -1)
    
    mp_drawing.draw_landmarks(
        img,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    img = cv2.resize(img, (384*2,288*2))
    cv2.imwrite(FRAME_PAINT_DIRECTORY+"/"+path+"_Frame_"+str(count)+".jpeg", img)
    
    if count % n_plot == 0:
        cv2.imshow('image',img)
        cv2.waitKey(0)
    
    
def saving_csv(path,data):
    
    
    data_l = len(data)
    
    if data_l >= 412:
        df = pd.DataFrame(data, columns = ['x','y','z'])
        df.to_csv(SAVING_DIRECTORY+"/"+path+".csv",index=False,header=True, sep='\t')
    else:
        for i,landmarks in enumerate(data):
            print(i)
            if i == 0:
                df = pd.DataFrame(landmarks, columns = ['x','y','z'])
                df.to_csv(SAVING_DIRECTORY+"/"+path+"_pose_landmarks.csv",index=False,header=True, sep='\t')
            elif i == 1:
                df = pd.DataFrame(landmarks, columns = ['x','y','z'])
                df.to_csv(SAVING_DIRECTORY+"/"+path+"_face_landmarks.csv",index=False,header=True, sep='\t')
            

def saving_Video(path,count,fps):
    
    """
    https://stackoverflow.com/questions/43048725/python-creating-video-from-images-using-opencv
    https://stackoverflow.com/questions/72249011/appending-video-frame-to-list-using-opencv-take-a-lot-of-resources-in-ram
    https://stackoverflow.com/questions/73609006/how-to-create-a-video-out-of-frames-without-saving-it-to-disk-using-python?noredirect=1&lq=1
    """
    frames = []
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    
    print("Anzahl der Frames: "+str(count))
    
    for i in range(1,count):
        frames.append(cv2.imread(FRAME_PAINT_DIRECTORY+"/"+path+"_Frame_"+str(i)+".jpeg"))
    
    
    height,width,layers=frames[1].shape
    
    # Muss überlegt werden ob man die fs durch 1.25 rechnen will 
    out = cv2.VideoWriter(SAVING_V_DIRECTORY+"/"+path+"_Test6_.mp4",fourcc,fps,(width,height))
    
    for frame in frames:
        #img_result = MyImageTreatmentFunction(frame) # returns a numpy array image
        out.write(frame)
    
    cv2.destroyAllWindows()
    out.release()

def saving_error(path,error_l):
    
    with open(ERROR_DIRECTORY+"/"+path+"_errors_for_Frames.txt", "w") as text_file:
        
        for error in error_l:
            
            text_file.write(error+"\n")

def sorted_alphanumeric(data):
    
    """
    Sorts alphanumerically.
    :param data: data to be sorted e.g. a list
    :return: sorted data alphanumerically e.g. "User_2" before "User_10"
    """
    
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(data, key=alphanum_key)

def timer(starttime,lasttime):
    """
    Input: starttime = die Anfangszeit
           lasttime = die Endzeit
    Output: Differenz zwischen Start und Endzeit 
            --> Berechnung der benötigten Zeit für den Durchlauf einer Funktion
    """        
    totaltime = round((lasttime-starttime), 2)
    print("Total Time: "+str(totaltime))
    
    #return round((starttime-lasttime), 2)

def main():
    
    """
    Ideen:
        - Das Modell von Facemesh und Posedetection vllt nicht bei jedem Frame berechnen, weil es sonst in dem zusammengeschnitten Video so
          aussieht als würde es Zittern und es bewegt sich obwohl sich die Person nicht bewegt
          --> Zum andern müssen wir es für Jeden Frame berechnen, da man sonst vllt die Mikroexpressionen im Gesicht detectieren kann.
          --> Man benötigt eigentlich mindestens eine 140 HZ Kamera damit 140 Bilder pro Sekunde aufgenommen werden können
          --> Da Mikroexpressionen innerhalb von 40 bis 60 ms Stattfinden
          --> Für die Posedetection sind auch nur 25 Hz ausreichend
        - ausprobieren ob ich bei der CSV-Datei alles x in ein spalte und die y und z die die Spalten danach speichern kann
        
        
        - (Dynamisch Ordner erstellen für jedes Video damit darin alle Frames Gesepeichert werden)--> Nicht undbedingt nötig 
          wegen der Benennung
    
    Probleme:
    - Kann das dritte Video den ersten Frame anscheinend nicht bearbeiten und wirft deswegen einen Error
    - Wenn es nichts auslesen kann wirft es einen error muss mit einen Try and Catch umgangen werden
    - Oder mann mnuss darauf achten das in jeden frame die pose oder das gesicht, die Hand erkennbar sind
    
    - Was bedeutet die Z-Dimension für jeden einzelnen Frame? Bilder sind normalerweise 2-Dimensional!
      --> Ist das vllt die Tiefe des Bildes bzw. der Kamerawinkel? -> So wird das Bild als ein Dreidimensionaler Raum aufgefasst.
    
    https://github.com/google/mediapipe/tree/master/mediapipe/python/solutions
    https://google.github.io/mediapipe/solutions/face_detection.html
    https://www.kaggle.com/code/brodielamont/test-opencv-video-and-mediapipe
    
    """
    
    if not os.listdir(DATA_DIRECTORY):
        print('No input video files are present. Please put your files in the '
              '"data" directory, rebuild the image and run the container.')
    
    for i, file in enumerate(sorted_alphanumeric(os.listdir(DATA_DIRECTORY))):
        
        # Die zwei steht für die Modell komplexität
        # Die Hundert steht für den Frame welcher geplotted werden soll, so wird jeder Hunderste frame geplotted
        # Kann manchmal nicht die pose erkennen und wirft dadurch ein Error (Attributerror)
        
        #starttime = time.time()
        #print(get_Data_pose(file,2,100,200))
        #lasttime = time.time()
        #timer(starttime,lasttime)
        
        # Zur auswahl eines bestimmten Videos 
        #print(get_Data_pose('trial_truth_010.mp4',2,100,1000))
        

        # Die eins steht für die Maximale Anzahle an Faces
        # Die Zweihundert steht für den Frame welcher geplotted werden soll, so wird jeder Zweihunderste frame geplotted
        # Kann manchmal nicht das Gesicht erkennen und wirft dadurch ein Error (TypeErrorerror)
        
        #starttime = time.time() 
        #print(get_Data_mesh(file,1,100,100))
        #lasttime = time.time()
        #timer(starttime,lasttime)
        
        # Zur auswahl eines bestimmten Videos 
        #print(get_Data_mesh('trial_truth_010.mp4',2,100,1000))
        
        # Die eins steht für die Anzahl der model selectionen
        # Die Hundert steht für den Frame welcher geplotted werden soll, so wird jeder Hunderste frame geplotted
        # Funktionniert ohne probleme und ohne error
        
        #starttime = time.time()
        #print(get_Data_detection(file,1,100,100)[0])
        #lasttime = time.time()
        #timer(starttime,lasttime)
        
        # Zur auswahl eines bestimmten Videos 
        #print(get_Data_detection('trial_truth_010.mp4',2,100,1000)[0])

        # Holistic kann Facemesh,pose und Handdetection gleichzeitig machen und hat alles in results drinne
        # funktioniert nur für Video 001 010 011,012,047,048,039
        # Gibt gewisse schwächen und erknnt auch nicht immer das Gesicht oder die pose deswegen musste ich einen try and catch bauen
        starttime = time.time()
        print(get_Data_holistic(file,2,100,1000))
        lasttime = time.time()
        timer(starttime,lasttime)
        
        # Zur auswahl eines bestimmten Videos
        print(get_Data_holistic('trial_truth_001.mp4',2,100,1000))


DATA_DIRECTORY = 'Clips/Truthful'
FRAME_DIRECTORY = 'Frame_Data'
ERROR_DIRECTORY = 'Error_Data'
SAVING_DIRECTORY = 'Detected_Data'
SAVING_V_DIRECTORY = 'Detected_V_Data'
FRAME_PAINT_DIRECTORY = 'Frame_P_Data'

main()