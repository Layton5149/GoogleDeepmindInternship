import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from micromlgen import port


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

# since one of the tasks has a unlimited time, only the first 5 tasks are known. to find time stamps for the rest of the tasks, this function is used to work backwards from the end time stamp
def calculateEndMinuteMarks(signal, fs, length):
    endMinuteStart = (len(signal) // fs // length) - 5
    endMinuteMarksArray = [endMinuteStart, endMinuteStart + 2, endMinuteStart + 3, endMinuteStart + 5]
    return endMinuteMarksArray

#function to split up csv files into segments, also used to get classifiers for stress (1 or 0), returns: a 2D array that contains segment arrays for each 10 seconds
def splitSignals(base_path, fs, length, file_type):
    endArray = []
    endClassifierArray = []

    segmentTranslator = fs*length # amount of hz per segment 
    for i in range(1,30):
        file_path = (f"{base_path}{i}/{file_type}")
        signal = pd.read_csv(file_path).squeeze().to_numpy()

        #filter PPG signal with butter filter
        if file_type == "BVP.csv":
            signal = bandpass_filter(signal, fs)

        # split into 60 second segments
        step = fs*length
        totalWindows = len(signal) // step


        trimmed_length = totalWindows * step
        trimmed_signal = signal[:trimmed_length]
        segmentArray = trimmed_signal.reshape(-1, step)

        endArray.append(segmentArray)

        endMinuteMarks = calculateEndMinuteMarks(signal, fs, length)

        for j in range(len(segmentArray)):
            if (j * segmentTranslator ) < (3*length*fs): # first 3 minutes - Rest
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
            elif (j * segmentTranslator - 1) <= (endMinuteMarks[3]*length*fs): # next 2 minutes - Rest
                endClassifierArray.append(0)

    return endArray, endClassifierArray

def bandpass_filter(signal, fs, lowcut=0.5, highcut=4.0, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
    filteredSignal = filtfilt(b, a, signal)
    return filteredSignal

# function to extract all EDA features, then return a dictionary with feature name to feature value pairs
def getEDAFeatures(signalSegmentArray, fs, length):
    pass

# function to extract all PPG features, then return a dictionary with feature name to feature value pairs
def getPPGFeatures(signalSegmentArray, file_path_base, fs, length):

    features = {
        "IBI Mean": [],
        "IBI STD": [],
        "RMSSD": [],
        "SD1": [],
        "SD2": [],
        "BVP Mean": [],
        "BVP STD": [],
        "BVP Skewness": [],
        "BVP Kurtosis": [],
    }

    for subjectSegments in signalSegmentArray:
        for segment in subjectSegments:

            # Basic BVP stats
            features["BVP Mean"].append(np.mean(segment))
            features["BVP STD"].append(np.std(segment))
            features["BVP Skewness"].append(skew(segment))
            features["BVP Kurtosis"].append(kurtosis(segment))


            # find peaks in BVP waves
            peaks, _ = find_peaks(segment, distance=fs*0.4)  # Assuming min 40 BPM

            # calculate IBI values
            peak_times = peaks / fs
            ibi = np.diff(peak_times)  # IBI in seconds

            if len(ibi) < 2:
                for k in ["IBI Mean", "IBI STD", "RMSSD", "SD1", "SD2",]:
                    features[k].append(np.nan)
                continue

            # IBI Features
            features["IBI Mean"].append(np.mean(ibi))
            features["IBI STD"].append(np.std(ibi, ddof=1))
            features["RMSSD"].append(np.sqrt(np.mean(np.diff(ibi) ** 2)))
            sd1 = np.sqrt(np.var(np.diff(ibi), ddof=1) / 2)
            sd2 = np.sqrt(2 * np.var(ibi, ddof=1) - sd1**2)
            features["SD1"].append(sd1)
            features["SD2"].append(sd2)

    return features
            

def getTempFeatures(signalSegmentArray, fs):

    features = {
        "temp_mean": [],
        "temp_std": [],
        "temp_min": [],
        "temp_max": [],
        "temp_range": [],
        "temp_median": [],
        "temp_skew": [],
        "temp_kurtosis": [],
        "temp_slope": [],
        "temp_deriv_mean": [],
        "temp_deriv_std": []
    } 

    for record in signalSegmentArray:
        for segment in record:
            derivative = np.diff(segment)
            time = np.arange(len(segment)) / fs
            
            # Linear trend (slope)
            slope = np.polyfit(time, segment, 1)[0]
            
            features["temp_mean"].append(np.mean(segment))
            features["temp_std"].append( np.std(segment))
            features["temp_min"].append(np.min(segment))
            features["temp_max"].append(np.max(segment))
            features["temp_range"].append(np.max(segment) - np.min(segment))
            features["temp_median"].append(np.median(segment))
            features["temp_skew"].append(skew(segment))
            features["temp_kurtosis"].append(kurtosis(segment))
            features["temp_slope"].append(slope)
            features["temp_deriv_mean"].append(np.mean(derivative))
            features["temp_deriv_std"].append(np.std(derivative))

    return features

def saveClassifierAsCCode(classifier):
    c_code = port(classifier)
    with open("random_forest_model.h", "w") as f:
        f.write(c_code)

def main():
    fs = 64
    length = 60 #60 second segments
    file_path_base = "Dataset/Data_29_subjects/Subjects/subject_"
    PPGSignalSegmentArray = splitSignals(file_path_base, fs, length, "BVP.csv")[0] # use this to extract features about each PPG segment

    #EDASignalSegmentArray = splitSignals(file_path_base, 4,length, "EDA.csv")[0] # use this to get features for each EDA segment
    #getEDAFeatures(EDASignalSegmentArray, 1, 1)

    TEMPsignalSegmentArray = splitSignals(file_path_base, 4, length, "TEMP.csv")[0]
    tempFeatureList = getTempFeatures(TEMPsignalSegmentArray, 4)

    print (len(tempFeatureList))


    featureList = getPPGFeatures(PPGSignalSegmentArray, file_path_base, fs, length)

    featureList.update(tempFeatureList)


    signalsegmentClassifierArray = splitSignals(file_path_base, fs, length, "BVP.csv")[1] # this is the classifier for stress (1 or 0) for each of the segments
    featureList["classifier"] = signalsegmentClassifierArray

    df = pd.DataFrame(featureList)
    df.dropna(axis=0, how='all', inplace=True)

    # Separate features and target
    y = df["classifier"]
    x = df.drop(columns=["classifier"])

    smote=SMOTE(sampling_strategy='minority') 
    x,y=smote.fit_resample(x,y)

    # Impute missing values (mean imputation)
    imputer = SimpleImputer(strategy="mean")
    x_imputed = imputer.fit_transform(x)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x_imputed, y, test_size=0.2, random_state=42)
    # Random Forest
    clf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
    clf.fit(x_train, y_train)
    # Predict and report
    y_pred = clf.predict(x_test)
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print (df.head(20))

    importances = clf.feature_importances_

    # Print them
    for feature, importance in zip(df.columns, importances):
        print(f"{feature}: {importance:.4f}")

    saveClassifierAsCCode(clf)

    


main()