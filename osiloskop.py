import sys, os, serial, serial.tools.list_ports, threading, csv, datetime, time
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from scipy.signal import find_peaks, butter, filtfilt


# --- Serial Reader ---
class SerialReader(QtCore.QObject):
    data_received = QtCore.pyqtSignal(int, int, int)
    error_signal = QtCore.pyqtSignal(str)

    def __init__(self, baud=115200):
        super().__init__()
        self.ser = None
        self.running = False
        self.baud = baud

        ports = list(serial.tools.list_ports.comports())
        if not ports:
            self.error_signal.emit("❌ Tidak ada port serial ditemukan")
            return

        try:
            self.ser = serial.Serial(ports[0].device, baud, timeout=0.05)
            print(f"✅ Terhubung ke {ports[0].device} @ {baud}")
            self.running = True
            self.thread = threading.Thread(target=self.read_serial, daemon=True)
            self.thread.start()
        except Exception as e:
            self.error_signal.emit(f"❌ Gagal buka serial: {e}")

    def read_serial(self):
        while self.running and self.ser:
            try:
                line = self.ser.readline().decode("utf-8").strip()
                if line:
                    parts = line.split(",")
                    if len(parts) == 3:
                        v1, v2, v3 = map(int, parts)
                        self.data_received.emit(v1, v2, v3)
            except Exception:
                continue

    def stop(self):
        self.running = False
        if self.ser:
            self.ser.close()


# --- BPM Detector ---
class BPMDetector:
    def __init__(self, fs=1000, window_size=5):
        self.fs = fs
        self.window_size = window_size
        self.peak_times = []
        self.b, self.a = butter(3, [20, 200], btype='band', fs=fs)
    
    def filter_signal(self, data):
        if len(data) < 30:
            return data
        try:
            return filtfilt(self.b, self.a, data)
        except:
            return data
    
    def detect_bpm(self, data, current_time):
        if len(data) < 100:
            return 0
        
        filtered = self.filter_signal(data)
        distance = int(0.3 * self.fs)
        peaks, _ = find_peaks(filtered, distance=distance, prominence=np.std(filtered) * 0.5)
        
        if len(peaks) > 0:
            last_peak_idx = peaks[-1]
            last_peak_time = current_time - (len(data) - last_peak_idx) / self.fs
            self.peak_times.append(last_peak_time)
            self.peak_times = [t for t in self.peak_times if current_time - t < self.window_size]
        
        if len(self.peak_times) >= 2:
            intervals = np.diff(self.peak_times)
            if len(intervals) > 0:
                avg_interval = np.mean(intervals)
                bpm = 60.0 / avg_interval if avg_interval > 0 else 0
                if 30 <= bpm <= 200:
                    return bpm
        return 0


# --- GUI ---
class PCGMonitor(QtWidgets.QMainWindow):
    def __init__(self, fs=1000):
        super().__init__()
        self.setWindowTitle("PCG Monitor - Auto Save 2 Minutes")
        self.resize(1000, 900)

        # Buffer data
        self.fs = fs
        self.N = 2000
        self.time = np.arange(self.N) / self.fs
        self.data1 = np.zeros(self.N)
        self.data2 = np.zeros(self.N)
        self.data3 = np.zeros(self.N)
        self.current_time = 0
        self.vref = 1.0
        self.start_time = time.time()
        self.duration = 120  # 2 menit
        self.rms_data = []   # simpan 1 data RMS per detik
        self.collecting = True

        # Layout utama
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.status = QtWidgets.QLabel("🔄 Mengumpulkan data selama 2 menit ...")
        layout.addWidget(self.status)

        # Plot
        self.plot1 = pg.PlotWidget(title="Channel A0")
        self.curve1 = self.plot1.plot(pen="r")
        layout.addWidget(self.plot1)

        self.plot2 = pg.PlotWidget(title="Channel A1")
        self.curve2 = self.plot2.plot(pen="g")
        layout.addWidget(self.plot2)

        self.plot3 = pg.PlotWidget(title="Channel A2")
        self.curve3 = self.plot3.plot(pen="b")
        layout.addWidget(self.plot3)

        # Serial
        self.reader = SerialReader()
        self.reader.data_received.connect(self.update_data)
        self.reader.error_signal.connect(self.show_error)

        # Timer
        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(30)

        self.rms_timer = QtCore.QTimer()
        self.rms_timer.timeout.connect(self.collect_rms)
        self.rms_timer.start(1000)  # setiap 1 detik

    def update_data(self, v1, v2, v3):
        self.data1 = np.roll(self.data1, -1); self.data1[-1] = v1
        self.data2 = np.roll(self.data2, -1); self.data2[-1] = v2
        self.data3 = np.roll(self.data3, -1); self.data3[-1] = v3
        self.current_time += 1.0 / self.fs

        # Cek apakah sudah 2 menit
        if self.collecting and (time.time() - self.start_time >= self.duration):
            self.collecting = False
            self.status.setText("💾 Menyimpan file CSV ...")
            self.save_rms_csv()
            self.save_raw_csv()
            self.status.setText("✅ Perekaman selesai dan file tersimpan di Downloads.")

    def update_plot(self):
        self.curve1.setData(self.time, self.data1)
        self.curve2.setData(self.time, self.data2)
        self.curve3.setData(self.time, self.data3)

    def collect_rms(self):
        if not self.collecting:
            return
        try:
            v1 = (self.data1 / 1023.0) * 5.0
            v2 = (self.data2 / 1023.0) * 5.0
            v3 = (self.data3 / 1023.0) * 5.0
            rms1 = np.sqrt(np.mean(v1**2))
            rms2 = np.sqrt(np.mean(v2**2))
            rms3 = np.sqrt(np.mean(v3**2))
            avg1, avg2, avg3 = np.mean(v1), np.mean(v2), np.mean(v3)
            self.rms_data.append([len(self.rms_data) + 1, avg1, rms1, avg2, rms2, avg3, rms3])
            self.status.setText(f"🟢 Merekam... ({len(self.rms_data)}/120 detik)")
        except Exception as e:
            self.status.setText(f"⚠️ RMS error: {e}")

    def get_download_path(self):
        return os.path.join(os.path.expanduser("~"), "Downloads")

    def save_rms_csv(self):
        try:
            folder = self.get_download_path()
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, datetime.datetime.now().strftime("pcg_rms_%Y-%m-%d_%H-%M-%S.csv"))
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Detik ke-", "A0 (Vavg)", "A0 (Vrms)", "A1 (Vavg)", "A1 (Vrms)", "A2 (Vavg)", "A2 (Vrms)"])
                writer.writerows(self.rms_data)
            print(f"💾 File RMS disimpan ke {filename}")
        except Exception as e:
            self.status.setText(f"❌ Gagal simpan RMS CSV: {e}")

    def save_raw_csv(self):
        try:
            folder = self.get_download_path()
            os.makedirs(folder, exist_ok=True)
            filename = os.path.join(folder, datetime.datetime.now().strftime("pcg_raw_%Y-%m-%d_%H-%M-%S.csv"))
            volt0 = (self.data1 / 1023.0) * 5.0
            volt1 = (self.data2 / 1023.0) * 5.0
            volt2 = (self.data3 / 1023.0) * 5.0
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Index", "A0 (ADC)", "A0 (Volt)", "A1 (ADC)", "A1 (Volt)", "A2 (ADC)", "A2 (Volt)"])
                for i in range(len(self.data1)):
                    writer.writerow([i, int(self.data1[i]), round(volt0[i],4),
                                     int(self.data2[i]), round(volt1[i],4),
                                     int(self.data3[i]), round(volt2[i],4)])
            print(f"💾 File RAW disimpan ke {filename}")
        except Exception as e:
            self.status.setText(f"❌ Gagal simpan RAW CSV: {e}")

    def show_error(self, msg):
        self.status.setText(msg)

    def closeEvent(self, event):
        self.reader.stop()
        event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PCGMonitor(fs=1000)
    win.show()
    sys.exit(app.exec_())
