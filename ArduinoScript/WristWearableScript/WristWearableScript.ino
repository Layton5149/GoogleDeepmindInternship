#include "MAX30105.h"    
#include "heartRate.h"   
#include "Wire.h"
#include "random_forest_model.h"

MAX30105 particleSensor;

void setup() {
  Serial.begin(9600);
  Serial.println("Starting sensor...");
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

  Eloquent::ML::Port::RandomForest model;
  int prediction = model.predict(features);
  Serial.println(prediction);

}

void loop() {
  Serial.println("hello world");

  delay(10); // adjust for sample rate
}
