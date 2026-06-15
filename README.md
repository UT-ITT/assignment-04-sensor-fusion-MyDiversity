[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/AktWbCri)
# assignment-04-CV-Sensor-Fusion

# AR Game

To play the game you require a Sheet of paper with 4 Aruco markers (one in each corner)

Running the program opens your devices camera (camera device 0)

If the program detects 4 aruco markers, it sets them as the game borders and the game starts

You can now use ur finger to pop the targets and increase your score

Try to only expose the Hand you want to play with to the camera image,

as otherwise the wrong hand might get recognised by the handtracking.

If the Hand does not get recognised, it might help to get it out of the camera frame and back in again.

The finger detection is implemented using googles mediapipe hand landmarks detection

# Sensor fusion

The implementation starts with the recognision of the board using 4 aruco markers, afterwards aruco marker 5 gets
recognised and its position is displayed with a red dot. Simultaneously the DIPPID data gets extracted, where the gravity is subtracted from the acceleration to only get the dynamic acceleration without gravity. Using this dynamic acceleration, a position prediction is calculated and mixed with the actual marker position, it is depicted as a green dot.

Using the left/right arrow keys, the alpha value can be adjusted. It determines the blend between prediction and actual marker position. A higher alpha value gives more weight to the prediction which smoothens the trajectory of the green dot and makes it more resistant to noise, but slower to follow sudden motion of the marker/red dot.
Lowering alpha results in a sharper trajectory, following the marker faster but making the prediction more sensitive to noise.

Button_1 can be used to reset the prediction to the actual marker position in case the prediction drifts too far
