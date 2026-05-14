# Deep Learning based driver monitoring system (activity & object recognition)

## Problem 
In recent years there has been a lot of focus on developing driver monitoring software for integration in passenger cars and other vehicles to facilitate better safety and other functions that improve the user experience. By studying a person’s posture and body movements, intelligent interior vehicle algorithms can draw conclusions about a person’s alertness, attention and focus. Tomorrow’s cabin sensing features will include detection of passenger position, safety belt status and forgotten objects, as well as enabling multimodal functionality such as deeper AI and mood recognition.So the car is able to seamlessly transfer control of the vehicle to an awake and able driver, call for help in a medical emergency, or offer to play the perfect song for the moment. </br>

## Solution Approach:

### By Computer Vision</br>

1.Identification of the driver in order to allow the vehicle to automatically restore its preferences and settings.  </br>                           

2.Activity recognition: </br>
   2.1 Deep learning model for recognition of continuous driver’s activity. </br>
   2.2 This includes activities such as driver talking on the phone, eating while driving the vehicle.  These activities will alert the system and make the driver more aware of the dangerous situation. </br>

3.Detecting levels of driver impairment: </br>
  3.1 Using a camera and microphone for detecting drowsiness, distraction, yawn, eye closure, and joy in real-time. </br>
  3.2 Monitor driver fatigue and alert him when potential drowsiness situation is detected. </br>
  3.3 Monitor driver attentiveness by ensuring he’s keeping his eyes on the road and that he is aware of any dangerous situation. </br>
  3.4 Pilot a user interface thanks to the eyes by automatically selecting HMI areas. </br>

4.Hand gesture control: </br>
    A trained neural network to detect hand gestures for volume/ channel control or any other in built functionality of the car. </br>

5.An intelligent steering wheel that monitors the heart rate to detect potential drowsiness of the driver. This is made by embedding the hand-grip heart rate monitor into the steering wheel of the vehicle. </br>

6.After detection of the driver’s fatigue while driving, the driver can be alerted with special sensors or an electric impulse bracelet. </br>

### Driving Style Classifier-AI:
The Driving style is simply analyzed by computational methodologies (Artificial Intelligence) and applied computing of transportation. </br>

##### Implementations of all Classifications Using Fuzzy Logic model
Fuzzy Logic Model-A branch of Artificial Intelligence (AI), which will characterize the uncertainty in the data by adding truth and false concepts from common logic to a machine-generated model. </br>

##### Aggressive Driving Style Criteria: (Input Variables)

1.Sudden Accelerations or Decelerations </br>
2.Sudden Braking </br>
3.Sharp Turns </br>
4.Set of events like start, stop, speed and turns </br>
5.Maximum and minimum rpm of the engine  </br>
6.Number of Red light Jumps</br>
7.Number of Tailgating cases</br>
8.Number of Aggressive Honking </br>
9.Number of Wrong side Overtaking  </br>

##### Steps Involved
1.Fuzzification: </br>
    This stage defines the membership functions and linguistic variables of the inputs.

2.Rules Evaluation: 
    In this stage, we will apply the fuzzy logic rules to calculate the output. </br>

3.Defuzzification:  
   The final conversion of the inputs to crisp results. </br>

##### Threshold Criteria:
(to be set for all classifications)
Example- the Threshold for harsh accelerations and decelerations has to be decided in by the System In accordance with the type of Road it is running on For example a city road or a state Highway or a National Highway. Because the speed limits will be different. </br>

### Novelty:
Our solution is a combination of three different approaches, which increases accuracy. This is a more intuitive use of the new generation of driver assistance functions. 
Most solutions being tested are based on just image processing. Combining computer vision, driving style, and heartbeat analysis and testing them has not been tried before. 
</br>

### Implementation Plan:
Implementation is divided into the following parts: </br>

1.Data acquisition :
A camera module is attached in front of the user which is continuously monitoring the activity of the driver. The hand-grip heart rate sensor embedded onto the steering wheel gives us the bpm of the driver. </br>

2.Pre-processing :
Filtering on sensor data. </br>

3.Data processing/ Feature Extraction:
AI/ Deep learning-based image processing for detecting the activity of the driver. </br>

4.Classification:
Fuzzy classifier to classify the driver’s state by scaling drowsiness, distraction, yawn, eye closure, and joy in real-time based on the threshold values. </br>

# Usage:
```
git clone https://github.com/prasad-kumkar/ai-driver-safety.git
```
Download models, haarcascade files and shape predictor file and put it in main folder.
Install required libraries:
- opencv-python
- dlib 
- keras
- imutils
- tensorflow
- keras 
- numpy
- pygame
``` pip install opencv-python dlib keras imutils numpy pygame
```
## Face landmarks:
Use of deep learning library **dlib's facial landmark predictor**
![WhatsApp Image 2020-01-12 at 20 23 16](https://user-images.githubusercontent.com/6639329/72222451-ce3bb580-358a-11ea-8211-3e8fbb8c4209.jpeg)


## Eye blinking counter
Detection of eye blinks and if more frequent eye blinks are detected, then play an alarm.

![WhatsApp Image 2020-01-12 at 20 22 25](https://user-images.githubusercontent.com/6639329/72222428-96cd0900-358a-11ea-9737-4edba38e4258.jpeg)


## Drowsiness Detection
Use of opencv and dlib library to detect and 
Deep learning algorithm to detect the driver's drowsiness in real time and play alarm sound if eyes are being closed for more than given frames.
```
python drowsiness_detection.py
```
![WhatsApp Image 2020-01-12 at 20 20 15](https://user-images.githubusercontent.com/6639329/72222060-b3ffd880-3586-11ea-93c0-ca9426d26e80.jpeg)

![WhatsApp Image 2020-01-12 at 20 28 45](https://user-images.githubusercontent.com/6639329/72222394-3938bc80-358a-11ea-9ef9-475411b69c47.jpeg)

## Activity Recognition : Yawning
Detection of yawn by analysing facial landmarks of mouth and eyes.
```
python yawn.py
```
![WhatsApp Image 2020-01-12 at 20 26 43](https://user-images.githubusercontent.com/6639329/72222408-4f467d00-358a-11ea-99c9-237b0721bcfe.jpeg)

![WhatsApp Image 2020-01-12 at 20 27 40](https://user-images.githubusercontent.com/6639329/72222409-57062180-358a-11ea-9d9a-42e920dbf805.jpeg)

## Mood/Emotion recognition
Use of tensorflow library and training model to classify mood/emotion in real time.
```
python mood_recognition.py
```
![WhatsApp Image 2020-01-12 at 20 24 18](https://user-images.githubusercontent.com/6639329/72222368-f2e35d80-3589-11ea-9958-5f71e2bd6bb0.jpeg)

![WhatsApp Image 2020-01-12 at 20 24 45](https://user-images.githubusercontent.com/6639329/72222381-11495900-358a-11ea-8b1b-c82f9426d08e.jpeg)
