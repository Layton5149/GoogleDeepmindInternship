#include "MAX30105.h"    
#include "heartRate.h"   
#include "Wire.h"
#include "random_forest_model.h"

MAX30105 particleSensor;

Eloquent::ML::Port::RandomForest model;

float signalList[960] = {0};  // Initialize all to 0
int arrayIndex = 0;
float dcAverage = 80235;      // Start at first placeholder value
float alpha = 0.95; // For DC removal (IIR high-pass)

// placeholder ir values for testing 
float placeHolderArray[100] = {
  80235, 80240, 80250, 80263, 80278, 80291, 80301, 80304, 80299, 80287,
  80269, 80247, 80228, 80215, 80210, 80214, 80226, 80242, 80259, 80274,
  80286, 80293, 80297, 80295, 80288, 80276, 80259, 80240, 80222, 80210,
  80208, 80216, 80234, 80257, 80279, 80295, 80305, 80308, 80303, 80291,
  80273, 80252, 80231, 80215, 80208, 80212, 80226, 80245, 80264, 80280,
  80291, 80297, 80298, 80294, 80284, 80270, 80252, 80232, 80217, 80209,
  80211, 80224, 80244, 80264, 80281, 80292, 80297, 80297, 80292, 80281,
  80264, 80245, 80227, 80214, 80209, 80214, 80228, 80247, 80266, 80281,
  80291, 80296, 80296, 80291, 80281, 80266, 80248, 80230, 80217, 80211,
  80215, 80228, 80246, 80265, 80280, 80290, 80295, 80296, 80291, 80281
};
int in = 0;

void getPPGFeatures(float signal[]) {
  return;
}

void setup() {
  Serial.begin(9600);
  delay(2000);
  Serial.println("Starting sensor...");

  Serial.println("hello world");
  Serial.println("hello world");
  /*Wire.begin();

  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("MAX30102 not found!");
    while (1);
  }

  particleSensor.setup(60, 4, 2, 100, 411, 4096);
  Serial.println("Sensor initialized.");
  */

  float features[20] = {
    36.5,  // temp_mean
    0.3,   // temp_std
    36.2,  // temp_min
    36.8,  // temp_max
    0.6,   // temp_range
    36.5,  // temp_median
    0.1,   // temp_skew
    -0.8,  // temp_kurtosis
    0.05,  // temp_slope
    0.002, // temp_deriv_mean
    0.001, // temp_deriv_std

    800,   // IBI Mean (ms)
    50,    // IBI STD
    35,    // RMSSD
    0.12,  // SD1
    0.25,  // SD2
    0.75,  // BVP Mean
    0.05,  // BVP STD
    0.1,   // BVP Skewness
    -0.5,  // BVP Kurtosis
  };

  int prediction = model.predict(features);
  Serial.println(prediction);
  Serial.println("hello world");

  static float dcAverage = placeHolderArray[0];

}

void loop() {

  // Read new IR value
  if (1/*particleSensor.check() == true*/) {
    long rawIR = placeHolderArray[in]; // will be particleSensor.getIR()
    in = in + 1;

    // Ignore faulty readings
    if (rawIR < 50000) {
      return;
    }

    // 1️⃣ DC Removal (High-pass filter)
    dcAverage = alpha * dcAverage + (1 - alpha) * rawIR;
    float acComponent = rawIR - dcAverage;

    // 2️⃣ Optional: Low-pass filter (~4 Hz) using simple IIR
    static float prevFiltered = 0;
    float beta = 0.2;  // adjust for smoothing
    float filtered = beta * acComponent + (1 - beta) * prevFiltered;
    prevFiltered = filtered;

    // 3️⃣ Store in buffer for future processing (if needed)
    signalList[arrayIndex] = 1.0; //WILL BE UPDATED TO STORE IR READ VALUE (AFTER FILTERING)
    arrayIndex = arrayIndex + 1;

    Serial.println(filtered);
    
  }

  delay(100);
}
