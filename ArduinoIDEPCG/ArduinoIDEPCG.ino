// Kode Arduino Nano - Baca 3 ADC dan kirim Serial (CSV)

// Definisi pin ADC
int adcPin0 = A0;  
int adcPin1 = A1;  
int adcPin2 = A2;  

void setup() {
  Serial.begin(115200);  // Kecepatan serial
}

void loop() {
  // Baca nilai ADC dari 3 channel
  int val0 = analogRead(adcPin0);  
  int val1 = analogRead(adcPin1);  
  int val2 = analogRead(adcPin2);  

  // Kirim data ke serial dalam format CSV: val0,val1,val2
  Serial.print(val0);
  Serial.print(",");
  Serial.print(val1);
  Serial.print(",");
  Serial.println(val2);

  delay(1);  // Sampling ~1ms (1000 Hz)
}
