
from pathlib import Path
import face_recognition
import pickle
from collections import Counter
from PIL import Image, ImageDraw
import shutil
import cv2


BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

DTEC_ALG_CHOICES = ["hog", "cnn"]
DTEC_ALG_DEFAULT = DTEC_ALG_CHOICES[0]
DETEC_ALG = DTEC_ALG_CHOICES[0]

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
Path("training").mkdir(exist_ok=True)
Path("training_video").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def path_object_to_str(obj):  
    if isinstance(obj, Path):         
        return str(obj) 

def encode_known_faces(model: str = DTEC_ALG_DEFAULT, encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    names = []
    encodings = []
    
    #from pictures
    for filepath in Path("training").glob("*/*"):
        print(filepath)
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)
            print(name)
            print(len(encoding))

    #from videos
    for filepath in Path("training_video").glob("*/*"):
        name = filepath.parent.name
        print(filepath)

        # open video
        video_capture = cv2.VideoCapture(path_object_to_str(filepath))
        # Check if camera opened successfully
        if (video_capture.isOpened()== False): 
            print("Error opening video file")

        # Read until video is completed
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret == True:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
                face_locations = face_recognition.face_locations(image, model=model)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                for encoding in face_encodings:
                    names.append(name)
                    encodings.append(encoding)
                    print(name)
                    print(len(encoding))                    
            else:
                break
   
    #save encondings in a file
    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode = "wb") as f:
        pickle.dump(name_encodings, f)        
        print('encoding saved to file\n')


def encode_new_known_face(image_file, name, model: str = DTEC_ALG_DEFAULT, encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    loaded_encodings = None

    if encodings_location.is_file():
        with encodings_location.open(mode="rb") as f:
            loaded_encodings = pickle.load(f)
    else:
        loaded_encodings = {"names": [], "encodings": []}

    image = face_recognition.load_image_file(image_file)
    
    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for encoding in face_encodings:
        loaded_encodings["names"].append(name)
        loaded_encodings["encodings"].append(encoding)
        print(name)
        print(len(encoding))
    
    with encodings_location.open(mode = "wb") as f:
        pickle.dump(loaded_encodings, f)      
        print('encoding saved to file\n')

    has_saved = False
    print(Path(image_file).parent.name)
    for filepath in Path("training").glob("*/*"):
        print(filepath.parent)
        if filepath.parent.name == name:
            has_saved = True
            #save the image            
            shutil.copy(image_file, filepath.parent)    
            print("image saved")   
            break
    
    if has_saved == False:
        path = Path("training" + "/" + name +"/")
        path.mkdir(parents=True, exist_ok=True)
        shutil.copy(image_file, path)  
        print("image saved")   


def encode_new_known_face_video(video_file, name, model: str = DTEC_ALG_DEFAULT, encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    # open video
    video_capture = cv2.VideoCapture(video_file)
    # Check if camera opened successfully
    if (video_capture.isOpened()== False): 
        print("Error opening video  file")

    # Read until video is completed
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if encodings_location.is_file():
                with encodings_location.open(mode="rb") as f:
                    loaded_encodings = pickle.load(f)
            else:
                loaded_encodings = {"names": [], "encodings": []}
         
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                loaded_encodings["names"].append(name)
                loaded_encodings["encodings"].append(encoding)
                print(name)
                print(len(encoding))
            
            with encodings_location.open(mode = "wb") as f:
                pickle.dump(loaded_encodings, f)      
                print('encoding saved to file\n')
        else:
            break
    
    video_capture.release()

    has_saved = False
    print(Path(video_file).parent.name)
    for filepath in Path("training_video").glob("*/*"):
        print(filepath.parent)
        if filepath.parent.name == name:
            has_saved = True
            #save the image            
            shutil.copy(video_file, filepath.parent)    
            print("video saved")   
            break
    
    if has_saved == False:
        path = Path("training_video" + "/" + name +"/")
        path.mkdir(parents=True, exist_ok=True)
        shutil.copy(video_file, path)  
        print("video saved")   


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)

    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    if votes:
        return votes.most_common(1)[0][0]
    


def _display_face(draw, bounding_box, name):

    top, right, bottom, left = bounding_box

    draw.rectangle(((left, top), (right, bottom)), outline = BOUNDING_BOX_COLOR)

    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)

    draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill="blue", outline="blue")

    draw.text((text_left, text_top), name, fill="white")



def recognize_faces(image_location: str, model: str = DTEC_ALG_DEFAULT, 
                    encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    
    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        print(name, bounding_box)
        _display_face(draw, bounding_box, name)
    
    del draw
    pillow_image.show()


def recognize_faces_video(video_file: str, model: str = DTEC_ALG_DEFAULT, 
                    encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    
    print('recognize_faces_video: start')
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # open video
    video_capture = cv2.VideoCapture(video_file)
    # Check if camera opened successfully
    if (video_capture.isOpened()== False): 
        print("Error opening video  file")

    # Read until video is completed
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        if ret == True:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for unknown_encoding in face_encodings:
                name = _recognize_face(unknown_encoding, loaded_encodings)
                
                if not name:
                    name = "Unknown"
                
                print(name)
        else:
            break
    
    print('recognize_faces_video: end')   
    


def validate(model: str = DTEC_ALG_DEFAULT):
    for filepath in Path("validation").rglob("*"):
        if filepath.is_file():
            recognize_faces(image_location=str(filepath.absolute()), model=model)

encode_known_faces(model = DETEC_ALG)

#encode_new_known_face('output/123.jpg', 'anayah', model = DETEC_ALG)

#encode_new_known_face_video('shot.mp4', 'hafeez', model = DETEC_ALG)

recognize_faces("validation/ah.jpg", model = DETEC_ALG)

#recognize_faces_video('shot.mp4', model = DETEC_ALG)

#validate(model = DETEC_ALG)   