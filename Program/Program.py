import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image as PilImage
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image as KivyImage
from kivy.core.window import Window

model = load_model('model.keras')

label_map = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Sad',
    5: 'Surprise',
    6: 'Neutral'
}


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

Window.size = (800, 500)

class EmotionApp(App):
    def build(self):
        self.title = 'Emotion Detector'  # 
        self.img1 = KivyImage()
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)  
        self.label = Label(text="Press Capture button to detect emotion", font_size='20sp')  
        layout.add_widget(self.img1)
        layout.add_widget(self.label)
        btn = Button(text="Capture", size_hint=(.3, .3), pos_hint={'center_x': .5}, background_color=(.5, .5, .5, 1))  
        btn.bind(on_press=self.capture)
        layout.add_widget(btn)
        Clock.schedule_interval(self.update, 1.0/33.0)
        return layout

    def update(self, dt):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w+10, y+h+25), (0, 255, 0), 2)

            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture1 = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr') 
            texture1.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img1.texture = texture1

    def capture(self, instance):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h+25, x:x+w+10]
                cv2.imwrite("captured_face.jpg", face_crop)
            
            img = PilImage.open("captured_face.jpg").convert('L')
            img = img.resize((48, 48))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 48, 48, 1)
            prediction = model.predict(img_array)
            label_index = np.argmax(prediction)
            label = label_map[label_index]
            self.label.text = "Predicted emotion: " + label

if __name__ == '__main__':
    EmotionApp().run()