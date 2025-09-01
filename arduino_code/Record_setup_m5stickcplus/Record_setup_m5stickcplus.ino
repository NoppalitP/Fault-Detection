
#include <M5Unified.h>
///////config

int Board_rate = 500000;
int Sampling_rate = 22050;

/////
static constexpr size_t BLOCK_SIZE = 512;
int16_t buffer[BLOCK_SIZE];

// FFT variables for frequency analysis
float fft_real[BLOCK_SIZE];
float fft_imag[BLOCK_SIZE];

// Current page (0 = PCM sender, 1 = Monitor)
int currentPage = 0;

// Audio analysis variables
float currentDB = 0.0;
float dominantFreq = 0.0;
unsigned long lastUpdate = 0;

// Waveform display variables
int16_t waveformBuffer[128];  // Smaller buffer for display
int waveformIndex = 0;

// Sender page animation variables
unsigned long lastSenderUpdate = 0;
int animationFrame = 0;
bool isTransmitting = false;
unsigned long transmissionStart = 0;
int dataPacketsSent = 0;
float transmissionRate = 0.0;

void setup() {
  auto cfg = M5.config();
  M5.begin(cfg);
  
  Serial.begin(Board_rate);
  M5.Mic.begin();
  
  // Initialize display
  M5.Display.begin();
  M5.Display.setTextSize(2);
  M5.Display.setTextColor(WHITE);
  pinMode(10, OUTPUT);
  showCurrentPage();
}

void loop() {
  M5.update();
  
  // Handle button press for page switching
  if (M5.BtnA.wasPressed()) {
    // Play button sound and flash LED
    playButtonFeedback();
    
    currentPage = (currentPage + 1) % 2;
    M5.Display.clear();
    showCurrentPage();
  }
  
  // Record audio data
  if (M5.Mic.record(buffer, BLOCK_SIZE, Sampling_rate)) {
    if (currentPage == 0) {
      // Page 0: Send PCM data via Serial
      sendPCMData();
    } else {
      // Page 1: Monitor dB and frequency
      analyzeAudio();
      updateMonitorDisplay();
    }
  }
}

void sendPCMData() {
  // === SEND HEADER ===
  Serial.write(0xAA);
  Serial.write(0x55);

  // === SEND PCM DATA ===
  Serial.write((uint8_t*)buffer, sizeof(buffer));

}

void playButtonFeedback() {
  // Play beep sound (if speaker available)
  M5.Speaker.tone(10000, 400); 
  
  // Flash built-in LED
digitalWrite(10, LOW);
delay(100);
digitalWrite(10, HIGH);
delay(100);
digitalWrite(10, LOW);
  // Alternative: Use display flash
  M5.Display.fillScreen(WHITE);
  delay(50);
  M5.Display.fillScreen(BLACK);
}

void analyzeAudio() {
  // Calculate RMS for dB level
  float mean = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) mean += buffer[i];
  mean /= BLOCK_SIZE;

  // 2. คำนวณ RMS
  double sumSq = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    float x = buffer[i] - mean;
    sumSq += x * x;
  }
  float rms = sqrt(sumSq / BLOCK_SIZE);

  // 3. dBFS และแปลงเป็น dB SPL
  float dbFS = 20.0 * log10(rms / 32767.0);
  currentDB = dbFS + 116.0 +54-80;        // calibration offset จากสเปค

  // 4. ไม่ให้ลงต่ำกว่า 0
  if (currentDB < 0) currentDB = 0;
  
  // Update waveform buffer (downsample for display)
  for (int i = 0; i < 128 && i * 4 < BLOCK_SIZE; i++) {
    waveformBuffer[i] = buffer[i * 4]; // Take every 4th sample
  }
  
  // Simple FFT for dominant frequency detection
  findDominantFrequency();
}

void findDominantFrequency() {
  // Copy buffer to real part, zero imaginary
  for (int i = 0; i < BLOCK_SIZE; i++) {
    fft_real[i] = buffer[i];
    fft_imag[i] = 0.0;
  }
  
  // Simple DFT (for demonstration - you might want to use a proper FFT library)
  float maxMagnitude = 0.0;
  int maxIndex = 0;
  
  // Check frequencies up to Nyquist (11025 Hz for Sampling_rate sample rate)
  for (int k = 1; k < BLOCK_SIZE/4; k++) {
    float real_sum = 0.0;
    float imag_sum = 0.0;
    
    for (int n = 0; n < BLOCK_SIZE; n++) {
      float angle = -2.0 * PI * k * n / BLOCK_SIZE;
      real_sum += fft_real[n] * cos(angle);
      imag_sum += fft_real[n] * sin(angle);
    }
    
    float magnitude = sqrt(real_sum * real_sum + imag_sum * imag_sum);
    
    if (magnitude > maxMagnitude) {
      maxMagnitude = magnitude;
      maxIndex = k;
    }
  }
  
  // Convert bin to frequency
  dominantFreq = (float)maxIndex * 22050.0 / BLOCK_SIZE;
}

void showCurrentPage() {
  M5.Display.setTextColor(WHITE);
  M5.Display.setCursor(0, 10);
  
  if (currentPage == 0) {
    drawSenderInterface();
  } else {
    M5.Display.fillRect(0, 0, M5.Display.width(), 30, BLUE);
    M5.Display.setCursor(10, 8);
    M5.Display.println("AUDIO MONITOR");
    M5.Display.println("");
    M5.Display.println("");
    M5.Display.setTextColor(YELLOW);
    M5.Display.setCursor(10, 40);
    M5.Display.println("Press [A] to switch");
  }
}

void drawSenderInterface() {
  M5.Display.clear();
  
  // Title with decorative border
  M5.Display.fillRect(0, 0, M5.Display.width(), 30, BLUE);
  M5.Display.setTextColor(WHITE);
  M5.Display.setTextSize(1);
  M5.Display.setCursor(10, 8);
  M5.Display.println("PCM AUDIO SENDER");
  
  // Status indicators
  M5.Display.setTextColor(GREEN);
  M5.Display.setTextSize(1);
  M5.Display.setCursor(10, 40);
  M5.Display.println("STATUS: ACTIVE");
  
  // Connection info box
  M5.Display.drawRect(5, 55, M5.Display.width()-10, 60, WHITE);
  M5.Display.setTextColor(CYAN);
  M5.Display.setCursor(10, 65);
  M5.Display.println("SERIAL CONFIG:");
  M5.Display.setTextColor(WHITE);
  M5.Display.setCursor(10, 80);
  M5.Display.println("Baud Rate: " + String(Board_rate));
  M5.Display.setCursor(10, 95);
  M5.Display.println("Sample Rate: " + String(Sampling_rate));

  
  // Instructions
  M5.Display.setTextColor(YELLOW);
  M5.Display.setCursor(10, 130);
  M5.Display.println("Press [A] to Monitor");
}

void updateSenderDisplay() {
  // Update every 100ms for smooth animation
  if (millis() - lastSenderUpdate > 100) {
    
    // Clear dynamic content area
    M5.Display.fillRect(0, 150, M5.Display.width(), M5.Display.height()-150, BLACK);
    
    // Transmission indicator
    M5.Display.setTextColor(GREEN);
    M5.Display.setCursor(10, 155);
    M5.Display.print("TRANSMITTING ");
    


    
    // Data counter
    M5.Display.setTextColor(WHITE);
    M5.Display.setCursor(10, 170);
    M5.Display.printf("Packets Sent: %d", dataPacketsSent);
    
    // Transmission rate
    M5.Display.setCursor(10, 185);
    M5.Display.printf("Rate: %.1f pkt/sec", transmissionRate);
    
    // Signal strength bar (based on mic input level)
    drawSignalStrength();
    
    // Activity LED simulation
    drawActivityLED();
    
    animationFrame++;
    lastSenderUpdate = millis();
  }
}

void drawSignalStrength() {
  // Calculate signal level from current buffer
  long sum = 0;
  for (int i = 0; i < BLOCK_SIZE; i++) {
    sum += abs(buffer[i]);
  }
  float avgLevel = (float)sum / BLOCK_SIZE;
  int signalBars = map(avgLevel, 0, 3000, 0, 5);
  
  M5.Display.setTextColor(CYAN);
  M5.Display.setCursor(10, 205);
  M5.Display.print("Signal: ");
  
  // Draw signal bars
  for (int i = 0; i < 5; i++) {
    int barHeight = 8 + (i * 2);
    int barX = 55 + (i * 12);
    int barY = 215 - barHeight;
    
    if (i < signalBars) {
      uint16_t color = (i < 2) ? GREEN : (i < 4) ? YELLOW : RED;
      M5.Display.fillRect(barX, barY, 8, barHeight, color);
    } else {
      M5.Display.drawRect(barX, barY, 8, barHeight, DARKGREY);
    }
  }
}

void drawActivityLED() {
  // Simulated activity LED
  int ledX = M5.Display.width() - 30;
  int ledY = 160;
  
  // LED flashes when transmitting
  if (isTransmitting && (millis() - transmissionStart < 100)) {
    M5.Display.setTextColor(GREEN);
  } else {
    M5.Display.drawCircle(ledX, ledY, 8, DARKGREY);
    M5.Display.setTextColor(DARKGREY);
  }
  

  
  // Reset transmission flag
  if (millis() - transmissionStart > 100) {
    isTransmitting = false;
  }
}

void updateMonitorDisplay() {
  // Update display every 50ms for smoother waveform
  if (millis() - lastUpdate > 50) {
    // Clear only the data area
    M5.Display.fillRect(0, 60, M5.Display.width(), M5.Display.height()-60, BLACK);
    
    // Draw waveform
    drawWaveform();
    
    // Display audio info
    M5.Display.setTextSize(1);
    M5.Display.setTextColor(GREEN);
    M5.Display.setCursor(10, 155);
    M5.Display.print("METRICS ");
    M5.Display.setTextColor(WHITE);
    M5.Display.setCursor(10, 170);
M5.Display.printf("dB: %.1f", currentDB);
    M5.Display.setCursor(10, 185);
M5.Display.printf("Freq: %.0f Hz", dominantFreq);
    
    // Visual dB meter (horizontal bar)
    int barLength = map(constrain(currentDB, 0, 80), 0, 80, 0, M5.Display.width() - 20);
    M5.Display.drawRect(10, M5.Display.height() - 25, M5.Display.width() - 20, 8, WHITE);
    M5.Display.fillRect(11, M5.Display.height() - 24, barLength, 6, 
                       currentDB > 60 ? RED : (currentDB > 40 ? YELLOW : GREEN));
    
    M5.Display.setTextSize(2); // Reset text size
    lastUpdate = millis();
  }
}

void drawWaveform() {
  int displayWidth = M5.Display.width();
  int displayHeight = M5.Display.height();
  int waveHeight = 80; // Height allocated for waveform
  int waveTop = 70;    // Top Y position of waveform
  int centerY = waveTop + waveHeight / 2;
  
  // Draw center line
  M5.Display.drawLine(0, centerY, displayWidth, centerY, DARKGREY);
  
  // Draw waveform
  M5.Display.setColor(CYAN);
  
  for (int x = 1; x < displayWidth && x < 128; x++) {
    // Map audio sample to display coordinates
    int y1 = map(waveformBuffer[x-1], -5000, 5000, waveTop + waveHeight, waveTop);
    int y2 = map(waveformBuffer[x], -5000, 5000, waveTop + waveHeight, waveTop);
    
    // Constrain to display area
    y1 = constrain(y1, waveTop, waveTop + waveHeight);
    y2 = constrain(y2, waveTop, waveTop + waveHeight);
    
    // Draw line segment
    M5.Display.drawLine(x-1, y1, x, y2, CYAN);
  }
  
  // Draw waveform border
  M5.Display.drawRect(0, waveTop, displayWidth, waveHeight, WHITE);
}