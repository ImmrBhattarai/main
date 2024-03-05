import os
from keras.models import load_model
import numpy as np
import cv2

model = load_model("/model_vgg19.h5")

WRITER = None
(Width, Height) = (None, None)

output_video = '/output_video/demo_output.mp4'

file = os.listdir('/input_video')
directory = '/input_video'
files = os.listdir(directory)

classes = ['Card', 'Goal', 'None', 'Substitution']

for file in files:
    capture_video = cv2.VideoCapture(os.path.join(directory, file))
    frameRate = capture_video.get(cv2.CAP_PROP_FPS)


while True:
    ret, frame = capture_video.read()
    if not ret:
        break
    if Width is None or Height is None:
        (Width, Height) = (int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype('float32')
    frame = np.array(frame).reshape(224, 224, 3)
    preds = model.predict(np.expand_dims(frame, axis=0))[0]
    print(preds)
    i = preds.argmax(axis=0)
    label = classes[i]

    if WRITER is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        WRITER = cv2.VideoWriter(output_video, fourcc, frameRate, (Width, Height), True)

    if label != 'None':
        WRITER.write(output)

    print(label)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

print("Finalizing.....")
WRITER.release()
capture_video.release()