





NanoSense is an embedded AI system designed during my Artificial Intelligence Internship (June–August 2025) to monitor health signals in older adults using photoplethysmography (PPG) and Bluetooth Low Energy (BLE). The pipeline performs signal processing (Butterworth filtering, time-series segmentation), extracts HRV features (e.g., RMSSD, SDNN) along with skewness and kurtosis, and runs an optimized Random Forest classifier directly on an Arduino Nano 33 BLE Sense. Desktop evaluation reached 91% F1, and after model compression/embedded optimizations the on-device model achieved 84% F1 in real-time tests. The repository also includes an ACM-format paper documenting the system design, methodology, and results.   

Highlights   

End-to-end pipeline: PPG capture → Butterworth filter → segmentation → feature engineering (HRV + statistical) → Random Forest inference → BLE telemetry.   

Embedded first: Memory/latency-aware feature set and compressed model tailored for the Nano 33 BLE Sense.   

Connectivity: BLE beacons for context and low-energy data streaming.   


Reproducible research: ACM-style paper and experiment logs.      

   
| Setting             | Metric   | F1-Score |
| ------------------- | -------- | -------: |
| Desktop (Python)    | Macro F1 | **0.91** |
| On-device (Arduino) | Macro F1 | **0.84** |
