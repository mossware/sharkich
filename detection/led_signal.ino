const int redPin = 11;
const int greenPin = 10;
const int bluePin = 9;
//using rgb led module KY-016
int detectionState = 0;  // 0 = no drone, 1 = some detection, 2 = consistent detection

void setup() {
  Serial.begin(115200);  // Start serial communication
  pinMode(redPin, OUTPUT);
  pinMode(greenPin, OUTPUT);
  pinMode(bluePin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    char incomingByte = Serial.read();  
    
    if (incomingByte == '1') {
      detectionState = 1;
    } else if (incomingByte == '0') {
      detectionState = 0;
    } else if (incomingByte == '2') {
      detectionState = 2;
    }
  }

  switch (detectionState) {
    case 0:
      // No drone detected, show green
      analogWrite(redPin, 0);
      analogWrite(greenPin, 255);
      analogWrite(bluePin, 0);
      break;
      
    case 1:
      // At least one detected, show orange
      analogWrite(redPin, 230);
      analogWrite(greenPin, 100);  
      analogWrite(bluePin, 0);
      break;
      
    case 2:
      // Consistent drone detected, red 
      analogWrite(redPin, 255);
      analogWrite(greenPin, 0);
      analogWrite(bluePin, 0);
      break;
  }
}
