import sys, serial, serial.tools.list_ports, threading, csv, datetime
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
        self.setWindowTitle("PCG Monitor - Arduino Nano (3 Channel) + BPM Detection + CSV Export")
        self.resize(1000, 950)

        # Buffer data
        self.fs = fs
        self.N = 2000
        self.time = np.arange(self.N) / self.fs
        self.data1 = np.zeros(self.N)
        self.data2 = np.zeros(self.N)
        self.data3 = np.zeros(self.N)
        self.current_time = 0
        self.autoscale = False
        
        # BPM Detectors
        self.bpm_detector1 = BPMDetector(fs)
        self.bpm_detector2 = BPMDetector(fs)
        self.bpm_detector3 = BPMDetector(fs)

        # Layout utama
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        # --- Kontrol ---
        control_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(control_layout)

        # Button autoscale
        self.btn_autoscale = QtWidgets.QPushButton("🔄 Enable Autoscale")
        self.btn_autoscale.setCheckable(True)
        self.btn_autoscale.clicked.connect(self.toggle_autoscale)
        control_layout.addWidget(self.btn_autoscale)

        # Dropdown scale manual
        self.scale_combo = QtWidgets.QComboBox()
        self.scale_combo.addItems(["0 - 1023", "0 - 500", "0 - 255", "-512 - 512"])
        self.scale_combo.currentIndexChanged.connect(self.set_manual_scale)
        control_layout.addWidget(QtWidgets.QLabel("Manual Scale:"))
        control_layout.addWidget(self.scale_combo)

        # Tombol simpan CSV
        self.btn_save_rms = QtWidgets.QPushButton("💾 Save RMS CSV")
        self.btn_save_rms.clicked.connect(self.save_rms_csv)
        control_layout.addWidget(self.btn_save_rms)

        self.btn_save_raw = QtWidgets.QPushButton("💾 Save Raw CSV")
        self.btn_save_raw.clicked.connect(self.save_raw_csv)
        control_layout.addWidget(self.btn_save_raw)

        control_layout.addStretch()

        # --- BPM Display ---
        bpm_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(bpm_layout)
        self.bpm_label1 = QtWidgets.QLabel("❤️ A0: -- BPM")
        self.bpm_label1.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
        bpm_layout.addWidget(self.bpm_label1)
        self.bpm_label2 = QtWidgets.QLabel("❤️ A1: -- BPM")
        self.bpm_label2.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        bpm_layout.addWidget(self.bpm_label2)
        self.bpm_label3 = QtWidgets.QLabel("❤️ A2: -- BPM")
        self.bpm_label3.setStyleSheet("font-size: 16px; font-weight: bold; color: blue;")
        bpm_layout.addWidget(self.bpm_label3)

        # --- Plot ---
        self.plot1 = pg.PlotWidget(title="Channel A0"); self.curve1 = self.plot1.plot(pen="r"); self.plot1.setYRange(0, 1023)
        self.plot2 = pg.PlotWidget(title="Channel A1"); self.curve2 = self.plot2.plot(pen="g"); self.plot2.setYRange(0, 1023)
        self.plot3 = pg.PlotWidget(title="Channel A2"); self.curve3 = self.plot3.plot(pen="b"); self.plot3.setYRange(0, 1023)
        main_layout.addWidget(self.plot1)
        main_layout.addWidget(self.plot2)
        main_layout.addWidget(self.plot3)

        # --- Status ---
        self.status = QtWidgets.QLabel("🔄 Menunggu data ...")
        main_layout.addWidget(self.status)

        # --- Serial Reader ---
        self.reader = SerialReader()
        self.reader.data_received.connect(self.update_data)
        self.reader.error_signal.connect(self.show_error)

        # --- Timer ---
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.update_plot); self.timer.start(30)
        self.bpm_timer = QtCore.QTimer(); self.bpm_timer.timeout.connect(self.update_bpm); self.bpm_timer.start(1000)

    # --- Fungsi GUI ---
    def toggle_autoscale(self):
        self.autoscale = self.btn_autoscale.isChecked()
        self.btn_autoscale.setText("✅ Autoscale Aktif" if self.autoscale else "🔄 Enable Autoscale")

    def set_manual_scale(self):
        if self.autoscale: return
        pilihan = self.scale_combo.currentText()
        if pilihan == "0 - 1023": yr = (0, 1023)
        elif pilihan == "0 - 500": yr = (0, 500)
        elif pilihan == "0 - 255": yr = (0, 255)
        else: yr = (-512, 512)
        for p in [self.plot1, self.plot2, self.plot3]: p.setYRange(*yr)

    def update_data(self, v1, v2, v3):
        self.data1 = np.roll(self.data1, -1); self.data1[-1] = v1
        self.data2 = np.roll(self.data2, -1); self.data2[-1] = v2
        self.data3 = np.roll(self.data3, -1); self.data3[-1] = v3
        self.current_time += 1.0 / self.fs
        self.status.setText(f"✅ Data masuk: {v1}, {v2}, {v3}")

    def update_bpm(self):
        bpm1 = self.bpm_detector1.detect_bpm(self.data1, self.current_time)
        bpm2 = self.bpm_detector2.detect_bpm(self.data2, self.current_time)
        bpm3 = self.bpm_detector3.detect_bpm(self.data3, self.current_time)
        self.bpm_label1.setText(f"❤️ A0: {bpm1:.1f} BPM" if bpm1 > 0 else "❤️ A0: -- BPM")
        self.bpm_label2.setText(f"❤️ A1: {bpm2:.1f} BPM" if bpm2 > 0 else "❤️ A1: -- BPM")
        self.bpm_label3.setText(f"❤️ A2: {bpm3:.1f} BPM" if bpm3 > 0 else "❤️ A2: -- BPM")

    def update_plot(self):
        self.curve1.setData(self.time, self.data1)
        self.curve2.setData(self.time, self.data2)
        self.curve3.setData(self.time, self.data3)
        if self.autoscale:
            for p in [self.plot1, self.plot2, self.plot3]:
                p.enableAutoRange(axis=pg.ViewBox.YAxis)

    # --- Simpan CSV RMS ---
    def save_rms_csv(self):
        filename = datetime.datetime.now().strftime("pcg_rms_%Y-%m-%d_%H-%M-%S.csv")
        try:
            chunk_size = 20
            total_data = len(self.time)
            volt0 = (self.data1 / 1023.0) * 5.0
            volt1 = (self.data2 / 1023.0) * 5.0
            volt2 = (self.data3 / 1023.0) * 5.0

            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Segment ke-", "A0 (Vavg)", "A0 (Vrms)", "A1 (Vavg)", "A1 (Vrms)", "A2 (Vavg)", "A2 (Vrms)"])
                segment = 1
                for i in range(0, total_data, chunk_size):
                    end = min(i + chunk_size, total_data)
                    win0, win1, win2 = volt0[i:end], volt1[i:end], volt2[i:end]
                    avg0, vrms0 = np.mean(win0), np.sqrt(np.mean(np.square(win0)))
                    avg1, vrms1 = np.mean(win1), np.sqrt(np.mean(np.square(win1)))
                    avg2, vrms2 = np.mean(win2), np.sqrt(np.mean(np.square(win2)))
                    writer.writerow([segment, round(avg0,4), round(vrms0,4), round(avg1,4), round(vrms1,4), round(avg2,4), round(vrms2,4)])
                    segment += 1

            self.status.setText(f"💾 File RMS disimpan ke {filename}")
        except Exception as e:
            self.status.setText(f"❌ Gagal simpan CSV: {e}")

    # --- Simpan CSV RAW ---
    def save_raw_csv(self):
        filename = datetime.datetime.now().strftime("pcg_raw_%Y-%m-%d_%H-%M-%S.csv")
        try:
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
            self.status.setText(f"💾 File RAW disimpan ke {filename}")
        except Exception as e:
            self.status.setText(f"❌ Gagal simpan RAW CSV: {e}")

    def show_error(self, msg):
        self.status.setText(msg)

    def closeEvent(self, event):
        self.reader.stop()
        event.accept()


# --- Main ---
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    win = PCGMonitor(fs=1000)
    win.show()
    sys.exit(app.exec_())
