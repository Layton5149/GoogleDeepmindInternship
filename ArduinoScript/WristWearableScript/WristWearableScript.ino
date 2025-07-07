#include "random_forest_model.h"

Eloquent::ML::Port::RandomForest rf;

void setup() {
  Serial.begin(115200);
  Serial.println("RandomForest model ready.");
}

void loop() {
  // Example input: replace with real BVP sensor data
  float features[9] = {579.2, 21.5, 12.38, 6.08, 24.10, 0.903, 0.224, -0.941};

  // Run prediction
  int predictedClass = rf.predict(features);

  // Print result
  Serial.print("Predicted class: ");
  Serial.println(predictedClass);

  delay(1000); // Run every second
}
