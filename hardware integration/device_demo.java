#include <Adafruit_NeoPixel.h>

int sensormax = 0;
int sensormin = 1024;
int LED = 7;

Adafruit_NeoPixel pixels(7,9, NEO_GRB + NEO_KHZ800);
void setup() {
  pixels.begin();
  pixels.setBrightness(200);
  pinMode(11, OUTPUT);
  Serial.begin(9600);

  while(millis()<10000){
    int sensorvalue = analogRead(A0);

      if (sensorvalue > sensormax){
        sensormax = sensorvalue;
      }

      if (sensorvalue < sensormin){
        sensormin = sensorvalue;
      }
  }
  Serial.println("Sensor Min: ");
  Serial.println(sensormin);
  Serial.println("Sensor Max: ");
  Serial.println(sensormax);
}

void loop() {
  digitalWrite(11, HIGH);
  int inputs = analogRead(A0);
  // Serial.println(inputs);

  int range = sensormax - sensormin;
  int section = range/LED;

  int pixel = inputs / section;
  pixel = constrain (pixel,0,LED -1);

  int red = map(inputs,0,1023,255,0);
  int green = map (inputs,0,1023,0,255);
  int leftpixel = pixel + 1;
  pixels.clear();

  for (int i =0; i <= pixel; i++){
    pixels.setPixelColor(i,pixels.Color(red,green,0));
    for(int j =leftpixel; j<=6; j++){
      pixels.setPixelColor(j,pixels.Color(255,255,255));
    }
  }

  pixels.show();

}