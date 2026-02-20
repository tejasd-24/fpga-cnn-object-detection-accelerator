#!/usr/bin/env python3
"""
Real-time webcam detection using FPGA CNN accelerator.
C-accelerated readout + numpy-vectorized classification.
Target: 30+ FPS on PYNQ Z2.
"""

import numpy as np
import cv2
import time
import os
import sys
import json
import argparse
import threading
import ctypes
import subprocess
from http.server import HTTPServer, BaseHTTPRequestHandler

# ─── Paths ──────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))
BITSTREAM   = os.path.join(DIR, "lyr3_cnn.bit")
WEIGHTS_BIN = os.path.join(DIR, "fpga_weights", "weights.bin")
FC_W_FILE   = os.path.join(DIR, "fpga_weights", "fc_weight.npy")
FC_B_FILE   = os.path.join(DIR, "fpga_weights", "fc_bias.npy")
CLS_FILE    = os.path.join(DIR, "fpga_weights", "classes.json")
C_SRC       = os.path.join(DIR, "fast_readout.c")
C_LIB       = os.path.join(DIR, "fast_readout.so")
ARM_C_SRC   = os.path.join(DIR, "arm_cnn.c")
ARM_C_LIB   = os.path.join(DIR, "arm_cnn.so")

# ─── Constants ──────────────────────────────────────────────────
IMG = 128;  N_WT = 23184;  N_CH = 64;  FM = 256;  CH_OFF = 48
SH0, SH1, SH2 = 2, 4, 6
R_CTRL=0x00; R_STAT=0x04; R_CH=0x20; R_ADDR=0x24; R_DATA=0x28; R_SHIFT=0x28

NAMES = ['airplane','cat','zebra','bus','bicycle','donut']
COLORS = [(80,80,255),(80,220,80),(80,255,255),(255,120,80),(255,80,220),(230,230,80)]


# ============================================================
#  C Library
# ============================================================
def load_c_lib():
    if not os.path.exists(C_LIB) or os.path.getmtime(C_SRC) > os.path.getmtime(C_LIB):
        print("Compiling C library...", end=" ", flush=True)
        r = subprocess.run(['gcc','-shared','-fPIC','-O2','-o',C_LIB,C_SRC],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED:\n{r.stderr}"); return None
        print("done.")
    lib = ctypes.CDLL(C_LIB)
    lib.read_features_full.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.read_features_full.restype = None
    lib.read_features_sub.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_int]
    lib.read_features_sub.restype = None
    lib.start_and_wait.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.start_and_wait.restype = ctypes.c_int
    lib.open_devmem.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
    lib.open_devmem.restype = ctypes.c_void_p
    return lib


# ============================================================
#  Vectorized classify + bbox (NO Python loops!)
# ============================================================
def classify_vec(feat_flat, fc_w, fc_b, names):
    """
    Spatial bin pool + softmax using pure numpy (zero Python loops).
    feat_flat: (64, 256) uint8 → pool to (1024,) → classify.
    """
    # Reshape to (64, 4, 4, 4, 4): ch, row_bin, row_px, col_bin, col_px
    fm = feat_flat.astype(np.float32).reshape(N_CH, 4, 4, 4, 4) / 255.0
    # Mean over pixels within each bin → (64, 4, 4) → flatten → (1024,)
    pooled = fm.mean(axis=(2, 4)).reshape(-1)
    # Softmax
    logits = pooled @ fc_w.T + fc_b
    e = np.exp(logits - logits.max())
    p = e / e.sum()
    i = int(np.argmax(p))
    return i, names[i], float(p[i]), p


def bbox_vec(feat_flat, cls_idx, fc_w):
    """
    CAM bounding box using numpy vectorization (minimal Python loops).
    """
    fm = feat_flat.astype(np.float32).reshape(N_CH, 16, 16)
    w = fc_w[cls_idx].reshape(N_CH, 4, 4)  # (1024,) → (64, 4, 4)

    # Mask saturated channels
    ch_means = fm.mean(axis=(1, 2))  # (64,)
    valid = ch_means <= 250  # (64,) bool

    # Expand weights to 16×16: each 4×4 bin has same weight
    # w: (64, 4, 4) → repeat to (64, 16, 16)
    w_exp = np.repeat(np.repeat(w, 4, axis=1), 4, axis=2)  # (64, 16, 16)

    # Zero out saturated channels
    w_exp[~valid] = 0

    # Weighted sum
    cam = (w_exp * fm).sum(axis=0)  # (16, 16)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    thr = max(np.percentile(cam, 70), 0.25)
    mask = cam > thr
    if mask.any():
        rows, cols = np.any(mask, 1), np.any(mask, 0)
        r1, r2 = np.where(rows)[0][[0, -1]]
        c1, c2 = np.where(cols)[0][[0, -1]]
        return (int(c1*8), int(r1*8), int(min(127,(c2+1)*8)), int(min(127,(r2+1)*8)))
    return (0, 0, 127, 127)


# ============================================================
#  Camera helpers
# ============================================================
def _reset_usb_camera():
    """Try to reset the USB camera device so V4L2 can re-open it."""
    import glob
    # Kill any process holding /dev/video*
    try:
        subprocess.run(['fuser', '-k', '/dev/video0'], capture_output=True, timeout=3)
    except:
        pass
    time.sleep(0.3)
    # Try USB device reset via usbdevfs
    try:
        import fcntl
        for dev in glob.glob('/dev/bus/usb/*/*'):
            try:
                USBDEVFS_RESET = 21780
                fd = os.open(dev, os.O_WRONLY)
                fcntl.ioctl(fd, USBDEVFS_RESET, 0)
                os.close(fd)
            except:
                try: os.close(fd)
                except: pass
    except:
        pass
    time.sleep(0.5)


class CameraThread:
    """Continuously reads frames in background. Main loop always gets latest.
    MJPEG format + V4L2 backend + USB reset for reliability."""
    def __init__(self, cam_idx=0, width=640, height=480):
        self._idx = cam_idx
        self._width = width
        self._height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._last_ok = time.time()

        # Try to open camera with retries + USB reset
        self.cap = None
        for attempt in range(5):
            self.cap = self._try_open()
            if self.cap is not None:
                break
            print(f"  Camera open failed (attempt {attempt+1}/5), resetting USB...",
                  flush=True)
            _reset_usb_camera()
        if self.cap is None:
            raise RuntimeError("Camera failed! Try: unplug+replug USB, then retry.")

        # Read first frame
        for _ in range(15):
            ret, f = self.cap.read()
            if ret and f is not None:
                self.frame = f
                self.h, self.w = f.shape[:2]
                break
            time.sleep(0.2)
        else:
            raise RuntimeError("Camera opened but no frames! Unplug+replug USB.")

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _try_open(self):
        """Try to open camera. Returns cap or None."""
        # Try V4L2 backend first
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(self._idx, backend)
            if cap.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Drain stale buffers
                for _ in range(5):
                    cap.grab()
                return cap
            cap.release()
        return None

    def _reader(self):
        while self.running:
            try:
                grabbed = self.cap.grab()
                if grabbed:
                    ret, f = self.cap.retrieve()
                    if ret and f is not None:
                        with self.lock:
                            self.frame = f
                        self._last_ok = time.time()
                        continue

                # Watchdog: reopen if stuck for 2 seconds
                if time.time() - self._last_ok > 2.0:
                    print("⚠ Camera stuck — resetting...", flush=True)
                    try: self.cap.release()
                    except: pass
                    _reset_usb_camera()
                    new_cap = self._try_open()
                    if new_cap is not None:
                        self.cap = new_cap
                    self._last_ok = time.time()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"⚠ Camera error: {e}", flush=True)
                time.sleep(0.5)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        try: self.cap.release()
        except: pass


# ============================================================
#  FPGA Engine
# ============================================================
class FPGAEngine:
    def __init__(self, clib):
        from pynq import Overlay, allocate
        self.allocate = allocate
        self.clib = clib

        print("Loading bitstream...", end=" ", flush=True)
        self.overlay = Overlay(BITSTREAM)
        print("done.")

        self.ip = None
        for n in dir(self.overlay):
            if n.startswith('_'): continue
            o = getattr(self.overlay, n)
            if hasattr(o, 'mmio') and 'cnn' in n.lower():
                self.ip = o; break
        if not self.ip: raise RuntimeError("CNN IP not found")

        self.mmio = self.ip.mmio if hasattr(self.ip, 'mmio') else None
        if self.mmio is None:
            from pynq import MMIO
            self.mmio = MMIO(self.ip.base_addr, 0x100)

        # C /dev/mem mapping (fallback for old bitstream)
        phys = self.ip.mmio.base_addr if hasattr(self.ip.mmio, 'base_addr') else self.ip.base_addr
        self.c_base = None
        if clib:
            self.c_base = clib.open_devmem(ctypes.c_uint32(phys), ctypes.c_uint32(0x100))

        self.dma = None
        for n in dir(self.overlay):
            if 'dma' in n.lower():
                o = getattr(self.overlay, n)
                if hasattr(o, 'sendchannel'):
                    self.dma = o; break

        # Check for DMA receive channel (new bitstream with readback)
        # Note: old DMA has recvchannel attribute but it's None
        self.has_dma_readback = (self.dma is not None and 
                                 hasattr(self.dma, 'recvchannel') and 
                                 self.dma.recvchannel is not None)

        # Load weights
        wt = np.fromfile(WEIGHTS_BIN, dtype=np.uint8)
        assert len(wt) == N_WT
        self.mmio.write(R_CTRL, 0x04); time.sleep(0.001)
        self.mmio.write(R_CTRL, 0x00)
        buf = self.allocate(shape=(N_WT,), dtype=np.uint8)
        buf[:] = wt
        self.dma.sendchannel.transfer(buf); self.dma.sendchannel.wait()
        del buf

        packed = (SH0&0x1F)|((SH1&0x1F)<<5)|((SH2&0x1F)<<10)
        self.mmio.write(R_SHIFT, packed)

        self._buf = self.allocate(shape=(IMG*IMG,), dtype=np.uint8)
        self._feat = np.zeros(N_CH * FM, dtype=np.uint8)

        # Pre-allocate DMA receive buffer if readback available
        if self.has_dma_readback:
            self._recv_buf = self.allocate(shape=(N_CH * FM,), dtype=np.uint8)
            print(f"FPGA ready. Readout: DMA (hardware)")
        elif self.c_base:
            print(f"FPGA ready. Readout: C-accel")
        else:
            print(f"FPGA ready. Readout: Python (slow!)")

    def run(self, gray128):
        """DMA→conv→readback. Returns (feat, conv_ms, read_ms)."""
        self._buf[:] = gray128.flatten().astype(np.uint8)
        # Flush CPU cache so DMA sends the CURRENT image
        if hasattr(self._buf, 'flush'):
            self._buf.flush()
        self.mmio.write(R_CTRL, 0x06); time.sleep(0.0001)
        self.mmio.write(R_CTRL, 0x02)
        self.dma.sendchannel.transfer(self._buf)
        self.dma.sendchannel.wait()

        # Start inference + wait
        t0 = time.time()
        if self.c_base:
            if self.clib.start_and_wait(self.c_base, ctypes.c_int(5000000)) != 0:
                raise TimeoutError("FPGA timeout")
        else:
            self.mmio.write(R_CTRL, 0x01)
            while True:
                if (self.mmio.read(R_STAT) >> 1) & 1: break
                if time.time() - t0 > 5.0:
                    raise TimeoutError("FPGA timeout")
        conv_ms = (time.time() - t0) * 1000

        # Feature readback
        t1 = time.time()
        if self.has_dma_readback:
            # ─── DMA readback (new bitstream) ───
            # Trigger readback FSM: bit 3 of control register
            self.dma.recvchannel.transfer(self._recv_buf)
            self.mmio.write(R_CTRL, 0x08)  # readback_start
            self.dma.recvchannel.wait()
            if hasattr(self._recv_buf, 'invalidate'):
                self._recv_buf.invalidate()
            self._feat[:] = np.array(self._recv_buf)
        elif self.c_base:
            # ─── C register readout (old bitstream) ───
            self.clib.read_features_full(
                self.c_base, self._feat.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(N_CH), ctypes.c_int(CH_OFF))
        else:
            # ─── Python fallback ───
            for ch in range(N_CH):
                self.mmio.write(R_CH, CH_OFF + ch)
                for a in range(FM):
                    self.mmio.write(R_ADDR, a)
                    _ = self.mmio.read(R_DATA)
                    self._feat[ch*FM+a] = self.mmio.read(R_DATA) & 0xFF
        read_ms = (time.time() - t1) * 1000

        return self._feat.reshape(N_CH, FM).copy(), conv_ms, read_ms


# ============================================================
#  ARM CNN C Library
# ============================================================
def load_arm_cnn_lib():
    """Compile and load the optimized C CNN inference library."""
    if not os.path.exists(ARM_C_SRC):
        print(f"WARNING: {ARM_C_SRC} not found, ARM mode will use slow numpy")
        return None
    if not os.path.exists(ARM_C_LIB) or os.path.getmtime(ARM_C_SRC) > os.path.getmtime(ARM_C_LIB):
        print("Compiling ARM CNN C library...", end=" ", flush=True)
        for flags in [
            ['-O3', '-mcpu=cortex-a9', '-mfpu=neon', '-mfloat-abi=hard'],
            ['-O3'],
        ]:
            r = subprocess.run(
                ['gcc', '-shared', '-fPIC'] + flags + ['-o', ARM_C_LIB, ARM_C_SRC],
                capture_output=True, text=True)
            if r.returncode == 0:
                print(f"done. (flags: {' '.join(flags)})")
                break
        else:
            print(f"FAILED:\n{r.stderr}")
            return None
    lib = ctypes.CDLL(ARM_C_LIB)
    lib.cnn_infer.argtypes = [ctypes.c_void_p]*4
    lib.cnn_infer.restype = ctypes.c_int
    return lib


# ============================================================
#  ARM Engine (C-optimized CNN, same architecture as FPGA)
# ============================================================
class ARMEngine:
    def __init__(self):
        self.wt = np.fromfile(WEIGHTS_BIN, dtype=np.uint8)
        self.shifts_arr = np.array([SH0, SH1, SH2], dtype=np.int32)
        self.output_buf = np.zeros(N_CH * FM, dtype=np.uint8)

        # Parse numpy kernels (fallback)
        self.kernels, off = [], 0
        for ic,oc,_ in [(1,16,128),(16,32,64),(32,64,32)]:
            n = oc*ic*9; raw = self.wt[off:off+n].astype(np.int8)
            k = np.zeros((oc,ic,3,3), dtype=np.int8); idx=0
            for ob in range(oc//16):
                for i in range(ic):
                    for c in range(16):
                        for w in range(9):
                            k[ob*16+c,i,w//3,w%3]=raw[idx]; idx+=1
            self.kernels.append(k); off+=n

        self.clib = load_arm_cnn_lib()
        if self.clib:
            print("ARM ready. (C-optimized, ~3.7 FPS)")
        else:
            print("ARM ready. (numpy fallback, ~0.3 FPS)")

    def run(self, gray128):
        img = gray128.flatten().astype(np.uint8)
        t0 = time.time()

        if self.clib:
            self.clib.cnn_infer(
                img.ctypes.data_as(ctypes.c_void_p),
                self.wt.ctypes.data_as(ctypes.c_void_p),
                self.shifts_arr.ctypes.data_as(ctypes.c_void_p),
                self.output_buf.ctypes.data_as(ctypes.c_void_p))
        else:
            self.output_buf[:] = self._numpy_infer(gray128)

        conv_ms = (time.time() - t0) * 1000
        return self.output_buf.reshape(N_CH, FM).copy(), conv_ms, 0.0

    def _numpy_infer(self, gray128):
        x = gray128.reshape(1,IMG,IMG).astype(np.uint8)
        for kern,sh in zip(self.kernels, [SH0,SH1,SH2]):
            ic,H,W = x.shape; oc = kern.shape[0]
            xp = np.zeros((ic,H+2,W+2),dtype=np.int32)
            xp[:,1:H+1,1:W+1] = x.astype(np.int32)
            out = np.zeros((oc,H,W),dtype=np.int32)
            ki = kern.astype(np.int32)
            for dy in range(3):
                for dx in range(3):
                    out += np.einsum('oi,ihw->ohw',ki[:,:,dy,dx],xp[:,dy:dy+H,dx:dx+W])
            x = np.clip(out>>sh,0,255).astype(np.uint8)
            x = x.reshape(oc,H//2,2,W//2,2).max(axis=(2,4))
        return x.flatten()


# ============================================================
#  MJPEG Stream
# ============================================================
class Stream(BaseHTTPRequestHandler):
    frame=None; lock=threading.Lock()
    def do_GET(self):
        if self.path=='/':
            self.send_response(200)
            self.send_header('Content-Type','text/html'); self.end_headers()
            self.wfile.write(b"""<!DOCTYPE html><html><head>
<title>FPGA CNN Live</title><style>
body{background:#0a0a1a;color:#fff;font-family:system-ui;text-align:center;margin:0;padding:20px}
h1{color:#0af;font-size:1.5em}img{border:2px solid #0af;border-radius:12px;max-width:95vw}
</style></head><body><h1>FPGA CNN &mdash; Live Detection</h1>
<img src="/stream"/><p style="color:#666;font-size:.9em">PYNQ Z2 | C-Accelerated</p>
</body></html>""")
        elif self.path=='/stream':
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace;boundary=f')
            self.end_headers()
            while True:
                try:
                    with Stream.lock: f=Stream.frame
                    if f is not None:
                        _,jpg=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,70])
                        self.wfile.write(b'--f\r\nContent-Type:image/jpeg\r\n\r\n')
                        self.wfile.write(jpg.tobytes()); self.wfile.write(b'\r\n')
                    time.sleep(0.02)
                except: break
        else: self.send_error(404)
    def log_message(self,*_): pass


# ============================================================
#  UI
# ============================================================
def draw(frame, idx, name, conf, probs, bbox, fps, conv_ms, read_ms, mode, names):
    h, w = frame.shape[:2]
    xo = (w-h)//2 if w > h else 0
    cw = min(w, h)
    s = cw / 128.0
    bx1,by1 = int(xo+bbox[0]*s), int(bbox[1]*s)
    bx2,by2 = int(xo+bbox[2]*s), int(bbox[3]*s)
    c = COLORS[idx%len(COLORS)]
    cv2.rectangle(frame,(bx1,by1),(bx2,by2),c,2)
    lbl = f"{name} {conf*100:.0f}%"
    (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
    cv2.rectangle(frame,(bx1,by1-th-10),(bx1+tw+6,by1),c,-1)
    cv2.putText(frame,lbl,(bx1+3,by1-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
    cv2.putText(frame,f"{mode} {fps:.1f} FPS",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.putText(frame,f"Conv:{conv_ms:.0f}ms Read:{read_ms:.0f}ms",(8,52),
                cv2.FONT_HERSHEY_SIMPLEX,0.45,(180,180,180),1)
    # Prob bars
    bx,by0,bmw,bh = w-150,18,110,15
    for i,n in enumerate(names):
        y=by0+i*(bh+4); p=float(probs[i])
        cv2.rectangle(frame,(bx,y),(bx+bmw,y+bh),(40,40,40),-1)
        bw=int(p*bmw)
        if bw>0: cv2.rectangle(frame,(bx,y),(bx+bw,y+bh),COLORS[i],-1)
        cv2.putText(frame,f"{n[:4]} {p*100:.0f}%",(bx+2,y+bh-2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.33,(255,255,255),1)


# ============================================================
#  Main
# ============================================================
ARM_FC_W = os.path.join(DIR, "fpga_weights", "arm_fc_weight.npy")
ARM_FC_B = os.path.join(DIR, "fpga_weights", "arm_fc_bias.npy")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['fpga','arm'], default='fpga')
    ap.add_argument('--port', type=int, default=5000)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--res', type=str, default='640x480',
                    help='Camera resolution WxH, e.g. 320x240')
    args = ap.parse_args()
    cam_w, cam_h = [int(x) for x in args.res.split('x')]

    print("="*60)
    print("  REAL-TIME OBJECT DETECTION — FPGA CNN ACCELERATOR")
    print("="*60)

    # Load classifier weights (ARM uses separate weights if available)
    if args.mode == 'arm' and os.path.exists(ARM_FC_W):
        fc_w, fc_b = np.load(ARM_FC_W), np.load(ARM_FC_B)
        print("Loaded ARM-specific classifier weights")
    else:
        fc_w, fc_b = np.load(FC_W_FILE), np.load(FC_B_FILE)
        if args.mode == 'arm':
            print("⚠ ARM classifier weights not found — using FPGA weights")
            print("  Run dump_arm_features.py + retrain_classifier.py first!")

    names = NAMES[:]
    if os.path.exists(CLS_FILE):
        with open(CLS_FILE) as f: names = json.load(f)
    print(f"Classes: {names}")

    clib = load_c_lib()
    if clib is None and args.mode == 'fpga':
        print("ERROR: gcc needed. apt install gcc"); sys.exit(1)

    engine = FPGAEngine(clib) if args.mode == 'fpga' else ARMEngine()

    # Threaded camera
    print(f"\nOpening camera {args.camera}...")
    cam = CameraThread(args.camera, width=cam_w, height=cam_h)
    print(f"Camera: {cam.w}×{cam.h}")

    # Stream
    srv = HTTPServer(('0.0.0.0', args.port), Stream)
    srv.socket.setsockopt(__import__('socket').SOL_SOCKET,
                          __import__('socket').SO_REUSEADDR, 1)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"\n★ Stream: http://192.168.2.99:{args.port}")
    print(f"Mode: {args.mode.upper()}")
    print("Ctrl+C to stop.\n")

    n, fps_s = 0, 0.0
    mode_lbl = "FPGA" if args.mode == 'fpga' else "ARM"

    try:
        while True:
            frame = cam.read()
            if frame is None: continue

            t0 = time.time()

            # Preprocess: center-crop to square, resize 128×128
            h, w = frame.shape[:2]
            if w > h:
                x1 = (w-h)//2; crop = frame[:, x1:x1+h]
            elif h > w:
                y1 = (h-w)//2; crop = frame[y1:y1+w, :]
            else:
                crop = frame
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)

            # Inference (FPGA or ARM — same CNN, same interface)
            feat, conv_ms, read_ms = engine.run(small)

            # Classify + bbox
            idx, name, conf, probs = classify_vec(feat, fc_w, fc_b, names)
            bbox = bbox_vec(feat, idx, fc_w)

            dt = time.time() - t0
            fps = 1.0/dt if dt > 0 else 0
            fps_s = 0.8*fps_s + 0.2*fps

            draw(frame, idx, name, conf, probs, bbox, fps_s, conv_ms, read_ms, mode_lbl, names)

            with Stream.lock:
                Stream.frame = frame.copy()

            n += 1
            if n % 20 == 0:
                top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
                s = " | ".join(f"{names[i]}:{probs[i]*100:.0f}%" for i in top)
                total_ms = dt * 1000
                print(f"\r  Frame {n} | {fps_s:.1f} FPS | total:{total_ms:.0f}ms conv:{conv_ms:.0f}ms read:{read_ms:.0f}ms | {s}   ",
                      end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\nDone. {n} frames.")
    finally:
        cam.release()


if __name__ == '__main__':
    main()
