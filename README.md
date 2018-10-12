### Speech Recognizer with Dynamic Time Warping

* A simple command recognizer to predict command for a given sample from a set of predefined commands. For a given sample, the command of the closest sample from training set found by Dynamic Time Warping is predicted as the corresponding command.


#### How to run 

* The recognizer could be run by the following command:

    
        python speech-recognizer-with-dtw.py --train <train-data-path> --test <test-data-path> --output <output-file>

    
    
* The arguments are optional. Their default values are as follows:

        --train: ./ProjectData/TrainData

        --test: ./ProjectData/TestData

        --output: ./predictions.txt
