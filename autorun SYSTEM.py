import Convert_to_images as CTI
import Create_blur_images as CBI
import Shuffle_blur
import Detect_Blur
import ANN_Pipeline
import Shuffle_data
import Digit_Classification
import Digit_Classification_TEST as DCT

def pause():
    programPause = raw_input("Press the <ENTER> key to continue...")


''' ### Comment out this part to not use blurred images ### '''

CTI.main()              # Use the data set provided to create jpg files.
pause()
CBI.main()              # Create blurred image pixel densities matrices (Now this data will be used for classification)
pause()
Shuffle_blur.main()     # Pre-process the blurred data before training the ANN.
pause()
Detect_Blur.main()      # Snippet for training ANN for blurred images.
pause()
ANN_Pipeline.main()     # Snippet to link the regression calculated data from ANN of blurred images to DC ANN.
pause()

''' ### Comment out till here ### '''

Shuffle_data.main()     # Pre-process the image data before training the ANN.
pause()
Digit_Classification.main()     # Snippet for training ANN for image classification.
pause()
DCT.main()              # Snippet to calculate the accuracy of the Classification.
