import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
#with open(file_path, 'r') as f:
    #lines = f.readlines()

# Extract start time (Unix timestamp) and sampling rate
#sampling_rate = 1  # Should be 1.0 for HR.csv

# Read HR values (each value is one second apart)
##hr_values = [float(line.strip()) for line in lines[2:]]

# Generate time axis: 0, 1, 2, ..., len(hr_values) - 1
#time_seconds = np.arange(len(hr_values))

def plotSegmentsSubplots(signal, fs, length):
    #THIS COULD BE REFINED
    step = fs*length
    totalWindows = len(signal) // step

    trimmed_length = totalWindows * step
    trimmed_signal = signal[:trimmed_length]
    segmentArray = trimmed_signal.reshape(-1, step)

    numOfRows = 5
    if (totalWindows % numOfRows != 0):
        numOfColumns = totalWindows // numOfRows + 1
    else:
        numOfColumns = totalWindows // numOfRows

    for i in range(1, totalWindows):
        plt.subplot(5, numOfColumns, i)
        plt.plot(np.arange(step) / fs / length, segmentArray[i-1], linewidth=0.5)
    
    plt.show()

def calculateEndMinuteMarks(signal, fs, length):
    endMinuteStart = (len(signal) // fs // length) - 5
    endMinuteMarksArray = [endMinuteStart, endMinuteStart + 2, endMinuteStart + 3, endMinuteStart + 5]
    return endMinuteMarksArray

# Convert minutes to seconds
start_minute_marks = [3, 13, 15, 20, 22, 25, 27]
#second_marks = []
#for m in minute_marks:
#    second_marks.append(m*60)

#import all hr stats
def splitSignals(base_path, fs, length, file_type):
    endArray = []
    endClassifierArray = []
    segmentTranslator = fs*length # amount of hz per segment 
    for i in range(1,21):
        file_path = (f"{base_path}{i}/{file_type}")
        signal = pd.read_csv(file_path).squeeze().to_numpy()

        # split into 60 second segments
        step = fs*length
        totalWindows = len(signal) // step

        trimmed_length = totalWindows * step
        trimmed_signal = signal[:trimmed_length]
        segmentArray = trimmed_signal.reshape(-1, step)

        endArray.append(segmentArray)

        endMinuteMarks = calculateEndMinuteMarks(signal, fs, length)

        for j in range(1, len(segmentArray)):
            if (j * segmentTranslator - 1) < (3*length*fs): # first 3 minutes - Rest
                endClassifierArray.append(0)
            elif (j * segmentTranslator - 1) < (13*length*fs): # next 10 minutes - Stress
                endClassifierArray.append(1)
            elif (j * segmentTranslator - 1) < (15*length*fs): # next 2 minutes- Rest
                endClassifierArray.append(0)
            elif (j * segmentTranslator - 1) < (20*length*fs): # next 5 minutes - Stress
                endClassifierArray.append(1)
            elif (j * segmentTranslator - 1) < (22*length*fs): # next 2 minutes - Rest
                endClassifierArray.append(0)
            elif (j * segmentTranslator - 1) < (25*length*fs): # next 3 minutes - stress
                endClassifierArray.append(1)
            elif (j * segmentTranslator - 1) < (27*length*fs): # next 2 minutes - Rest
                endClassifierArray.append(0)
            elif (j * segmentTranslator - 1) < (endMinuteMarks[0]*length*fs): # next x minutes - stress
                endClassifierArray.append(1)
            elif (j * segmentTranslator - 1) < (endMinuteMarks[1]*length*fs): # next 2 minutes - Rest
                endClassifierArray.append(0)
            elif (j * segmentTranslator - 1) < (endMinuteMarks[2]*length*fs): # next 1 minute - stress
                endClassifierArray.append(1)
            elif (j * segmentTranslator - 1) < (endMinuteMarks[3]*length*fs): # next 2 minutes - Rest
                endClassifierArray.append(0)

    return endArray, endClassifierArray


def main():
    fs = 64
    length = 60 # 60 seconds
    file_path_base = "Dataset/Data_29_subjects/Subjects/subject_"
    PPGSignalSegmentArray = splitSignals(file_path_base, fs, length, "BVP.csv")[0] # use this to extract features about each segment
    PPGsignalsegmentClassifierArray = splitSignals(file_path_base, fs, length, "BVP.csv")[1] # this is the classifier for stress (1 or 0) for each of the segments

    EDASignalSegmentArray = splitSignals(file_path_base, 4,length, "EDA.csv")[0]

    print (len(PPGSignalSegmentArray[6]))
    print (len(EDASignalSegmentArray[6]))

    #getPPGFeatures(PPGSignalSegmentArray)




main()