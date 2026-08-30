# Unlocked






## Inspiration
All of us have had close family members suffer from major strokes leaving them with significantly reduced motor movement and control. Unable to move or speak properly meant that communication was never truly through themselves, stripping the connection from them and a world that unites through an online world, especially with family members scattered across the world. Simple tasks such as hand-typing a WhatsApp message is a foreign due to the lack of fine motor control available in their hands. Through no fault of their own, they were isolated from those closest to them. This is why we created Unlocked - the virtual keyboard and mouse system that utilises a facial scanning system to produce keyboard and mouse inputs allowing someone to use a device without the need for hands.






## What does Unlocked do?
The app takes in a live video feed through a computer webcam and tracks 468 facial landmarks that are used to produce distinct gestures which are translated into unique keyboard and mouse inputs. These gestures are user-specific and calibrated upon app launch, though, we provide a recommended set of gestures that we believe results in the best ease of use. Through these facial gestures, a user can navigate a keyboard (through a D-Pad style selection), typing into anywhere possible such as a search bar, word document, or chat. Users can also operate a mouse cursor through face tilting, activating the equivalent of mouse buttons through gestures.





## How was Unlocked built?
Unlocked is a real-time computer vision pipeline buuilt from Python and other frameworks. Your browser streams webcam frames to a server over a persistent WebSocket. There, Google MediaPipe extracts four hundred sixty-eight facial landmarks and encodes them into an eleven-dimensional feature vector. A k-NN classifier, trained on your own calibration data, evaluates that vector. Once a prediction clears our confidence threshold and holds steady across several frames, Unlocked fires a real OS-level keystroke directly into whatever application has focus.


## System Structure

## System Structure

```txt
facial-gesture-keyboard/
│
├── static/
│   ├── index.html                  Main app shell served to the webview
│   ├── keyboard.html               On-screen keyboard layout and keys
│   ├── keyboard.js                 Key press handling and keyboard interaction logic
│   ├── toggle.html                 Toggle / control panel UI
│   ├── debug.html                  Debug view for landmarks, gestures, and focus state
│   ├── app.js                      Frontend logic, camera feed, and backend messaging
│   └── style.css                   Global styling for all views
│
├── main.py                         Application entry point and startup wiring
├── run.py                          Dev launcher for running the app locally
├── desktop_app.py                  Desktop window host and webview integration
│
├── gestures.py                     Facial landmark processing and gesture recognition
├── focus_watcher.py                UIA focus polling and typeable-element detection
├── windows.py                      Windows integration, keyboard visibility, mode switching
│
├── face_landmarker.task            MediaPipe face landmarker model file
├── calibration_data.json           Saved per-user gesture calibration thresholds
│
├── build.spec                      PyInstaller build specification
├── build/                          PyInstaller intermediate build output
├── dist/                           Packaged executable output
│
├── requirements.txt                Runtime Python dependencies
├── requirements-build.txt          Additional dependencies for building the executable
├── .gitignore                      Files and folders ignored by Git
└── README.md                       Project documentation
```
