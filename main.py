from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import cv2
import face_recognition
import os
from datetime import datetime
import csv

app = FastAPI()

path = 'Database'
images = []
classNames = []
myList = os.listdir(path)

print(myList)  # to check the data availiable

for cl in myList:  # getting the names from images
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):  # encoding images
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


def markAttendacne(name):  # marking attendance to csv file
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        namelist = []
        for line in myDataList:
            entry = line.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# markAttendacne('alibi')
encodeListKnown = findEncodings(images)
print("Encoding complete")

file = open(
    'Attendance.csv')
csvreader = csv.reader(file)
count = 0

clients = []


@app.websocket("/video-feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)

    try:
        while True:
            success, img = cap.read()
            if not success:
                break
            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurrFrame = face_recognition.face_locations(imgS)
            encodesCurrFrame = face_recognition.face_encodings(
                imgS, facesCurrFrame)

            for encodeFace, faceloc in zip(encodesCurrFrame, facesCurrFrame):
                matches = face_recognition.compare_faces(
                    encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(
                    encodeListKnown, encodeFace)
                # print(faceDis)
                matchIndex = np.argmin(faceDis)
                if matches[matchIndex]:
                    name = classNames[matchIndex].upper()
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2-35), (x2, y2),
                                  (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1+6, y2-6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendacne(name)
                else:
                    name = "None"
                    y1, x2, y2, x1 = faceloc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                                  (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', img)
            frame_bytes = buffer.tobytes()

            # Send the frame to the WebSocket client
            await websocket.send_bytes(frame_bytes)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
