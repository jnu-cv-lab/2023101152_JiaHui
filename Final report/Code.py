# -*- coding: utf-8 -*-
import os
import cv2
import time
import threading
import qrcode
import requests
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO
import insightface
from insightface.app import FaceAnalysis
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# ============ 全局配置 ============
os.environ["YOLO_VERBOSE"] = "False"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
all_sign_set = set()
QR_IMG_PATH = "qrcode.png"
attend_start_time = None
attend_end_time = None
all_student_list = []
need_refresh = False  # 新增：二维码签到后刷新标记
cam_sign_set = set()# 新增：累计识别到场人数集合(现场)
all_arrive_set = set()  # 累计所有到场人员：库内学生+陌生人（按人脸特征去重）

# ============ Flask 签到服务 ============
app = Flask(__name__)
sign_html = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>课堂签到</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
</head>
<body style="text-align:center;padding-top:50px">
    <h2>课堂统一签到</h2>
    <input id="name" placeholder="输入姓名" style="width:80%;padding:12px;font-size:16px"><br>
    <button onclick="submitSign()" style="margin-top:20px;padding:12px 30px;font-size:16px">签到</button>
    <p id="res"></p>
<script>
function submitSign(){
    let n=document.getElementById("name").value.trim();
    if(!n)return void(document.getElementById("res").innerText="请输入姓名");
    fetch("/sign",{
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body:JSON.stringify({name:n})
    }).then(r=>r.json()).then(d=>document.getElementById("res").innerText=d.msg);
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(sign_html)

@app.route("/sign", methods=["POST"])
def do_sign():
    global need_refresh
    name = request.get_json().get("name", "").strip()
    if not name:
        return {"code": 0, "msg": "姓名不能为空"}
    if name in all_sign_set:
        return {"code": 0, "msg": "已签到"}
    all_sign_set.add(name)
    need_refresh = True  # ? 签到成功，触发刷新
    return {"code": 1, "msg": "签到成功"}

def run_flask():
    #app.run(host="127.0.0.1", port=8088, debug=False, use_reloader=False)
    app.run(host="0.0.0.0", port=8088, debug=False, use_reloader=False)

# ============ 二维码本地生成+读取显示 ============
def generate_qr_to_file():
    max_retry = 10
    for i in range(max_retry):
        try:
            r = requests.get("http://127.0.0.1:4040/api/tunnels", timeout=5)
            public_url = r.json()["tunnels"][0]["public_url"]
            qr = qrcode.make(public_url)
            qr = qr.resize((500, 500), Image.Resampling.LANCZOS)
            qr.save(QR_IMG_PATH)
            print("? 二维码已保存到：", QR_IMG_PATH)
            return
        except Exception as e:
            print(f"第{i+1}次获取ngrok地址失败，重试中...")
            time.sleep(1)
    print("? 多次重试失败，请确认ngrok已运行: ngrok http 8088")

def show_qr_from_file():
    if not os.path.exists(QR_IMG_PATH):
        messagebox.showerror("错误", "二维码图片不存在，请先启动ngrok并重启程序")
        return
    try:
        img = Image.open(QR_IMG_PATH)
        img.thumbnail((900, 600))
        imgtk = ImageTk.PhotoImage(img)
        img_label.imgtk = imgtk
        img_label.configure(image=imgtk)
        print("? 已显示本地二维码图片")
    except Exception as e:
        messagebox.showerror("错误", "读取二维码图片失败：" + str(e))

# ============ 人脸识别初始化 ============
yolo = YOLO("yolov8n.pt", verbose=False)
face_app = FaceAnalysis(name='buffalo_l')
face_app.prepare(ctx_id=-1, det_size=(640, 640))

FEATURES_FILE = "/home/mjhyyfj/cv-course/src/bhw/face_features.npz"
if not os.path.isfile(FEATURES_FILE):
    raise FileNotFoundError("人脸特征库不存在")

data = np.load(FEATURES_FILE)
db_features = data["features"]
db_labels = data["labels"]
all_student_list = list(db_labels)
print(f"? 已加载人脸：{all_student_list}")

SIM_THRESHOLD = 0.25
OVERLAP_THRESH = 0.3

# ============ GUI ============
root = tk.Tk()
root.title("人脸识别+二维码双签到")
root.geometry("1400x750")

# 左侧
left_frame = tk.Frame(root, width=200, bg="#f0f0f0")
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
tk.Label(left_frame, text="控制面板", font=("Arial",14,"bold"), bg="#f0f0f0").pack(pady=5)

btn_open_qr = tk.Button(left_frame, text="显示签到二维码", command=show_qr_from_file, bg="#9C27B0", fg="white", width=15)
btn_open_qr.pack(pady=5)

btn_select_img = tk.Button(left_frame, text="选择图片", width=15)
btn_select_img.pack(pady=5)
lbl_path = tk.Label(left_frame, text="", wraplength=180, bg="#f0f0f0")
lbl_path.pack(pady=2)

tk.Label(left_frame, text="预期人数", bg="#f0f0f0").pack(pady=(10,0))
entry_total = tk.Entry(left_frame, width=10)
entry_total.insert(0, "40")
entry_total.pack(pady=2)

btn_img_detect = tk.Button(left_frame, text="图片考勤", bg="#4CAF50", fg="white", width=15)
btn_img_detect.pack(pady=8)

btn_camera_start = tk.Button(left_frame, text="开始考勤", bg="#2196F3", fg="white", width=15)
btn_camera_start.pack(pady=8)

btn_camera_stop = tk.Button(left_frame, text="结束考勤", bg="#f44336", fg="white", width=15, state=tk.DISABLED)
btn_camera_stop.pack(pady=8)

btn_export_log = tk.Button(left_frame, text="导出考勤日志", bg="#FF9800", fg="white", width=15)
btn_export_log.pack(pady=8)

# 中间画面区
right_frame = tk.Frame(root, bg="white")
right_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=5, pady=5)
img_label = tk.Label(right_frame, bg="gray")
img_label.pack(expand=True, fill=tk.BOTH)

# 右侧日志区
log_frame = tk.Frame(root, width=350, bg="#fafafa")
log_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
tk.Label(log_frame, text="考勤结果", font=("Arial",12,"bold"), bg="#fafafa").pack(pady=5)
txt_output = scrolledtext.ScrolledText(log_frame, width=40, height=35, font=("Consolas",10))
txt_output.pack(fill=tk.BOTH, expand=True)

# ============ 摄像头/人脸识别逻辑 ============
cap = None
camera_running = False
last_frame = None
process_lock = threading.Lock()
max_detected_count = 0
last_save_time = time.time()
SAVE_INTERVAL = 15
last_result = None

def stream_read_thread():
    global cap, camera_running, last_frame
    while camera_running:
        ret_grab = cap.grab()
        if not ret_grab:
            break
        _, frame = cap.retrieve()
        if frame is None:
            continue
        with process_lock:
            last_frame = frame.copy()

def analysis_frame(frame, rotate_90=True):
    global all_sign_set
    if rotate_90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    target_w, target_h = 720, 1280
    h_src, w_src = frame.shape[:2]
    scale = min(target_w / w_src, target_h / h_src)
    frame = cv2.resize(frame, (int(w_src*scale), int(h_src*scale)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]

    yolo_results = yolo(frame, verbose=False)
    persons = []
    for r in yolo_results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id == 0 and conf > 0.25:
                persons.append(list(map(int, box.xyxy[0])))

    faces = face_app.get(frame)
    front=middle=back=0
    for (x1,y1,x2,y2) in persons:
        cy=(y1+y2)/2
        if cy<h/3:front+=1
        elif cy<2*h/3:middle+=1
        else:back+=1
        best_face=None
        best_overlap=0
        for face in faces:
            fx1,fy1,fx2,fy2=face.bbox.astype(int)
            ix1,iy1,ix2,iy2=max(x1,fx1),max(y1,fy1),min(x2,fx2),min(y2,fy2)
            inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            face_area=(fx2-fx1)*(fy2-fy1)
            if face_area==0:continue
            overlap=inter/face_area
            if overlap>best_overlap and overlap>OVERLAP_THRESH:
                best_overlap=overlap
                best_face=face
        if best_face is not None:
            emb=best_face.normed_embedding
            sims=np.dot(db_features,emb)
            idx=np.argmax(sims)
            if sims[idx]>SIM_THRESHOLD:
                name=db_labels[idx]
                cam_sign_set.add(name)
                all_sign_set.add(name)
                all_arrive_set.add(name)  # 库内学生计入累计到场
                cv2.putText(frame,name,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            else:
                # 陌生人：用人脸特征哈希做唯一标识，同一个人多次出现只算一次
                stranger_id = f"陌生人_{hash(emb.tobytes()) % 100000}"
                all_arrive_set.add(stranger_id)  # 陌生人计入累计到场
                cv2.putText(frame,"未知",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,165,255),2)
        else:
            cv2.putText(frame,"未检测到人脸",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    total_person=len(persons)
    try:expected=int(entry_total.get())
    except:expected=40
    rate=(total_person/expected)*100 if expected>0 else 0
    return frame,total_person,expected,rate,front,middle,back

def show_image_to_gui(img):
    img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    im=Image.fromarray(img_rgb)
    im.thumbnail((900,600))
    imgtk=ImageTk.PhotoImage(im)
    img_label.imgtk = imgtk
    img_label.configure(image=imgtk)
#考勤日志
def update_log_text(detected,expected,rate,front,middle,back):
    txt_output.delete(1.0,tk.END)
    # 已签到总人数 = 人脸在册签到 + 二维码所有签到
    all_sign_num = len(all_sign_set)
    # 在册有效签到人数（只统计人脸库内的人员）
    valid_sign_num = len(set(all_sign_set) & set(all_student_list))
    # 出勤率 = 在册签到人数 / 左侧输入的应到人数
    sign_rate = (valid_sign_num / expected) * 100 if expected > 0 else 0
    absent_count = expected - valid_sign_num
    absent_list = list(set(all_student_list) - all_sign_set)

    txt_output.insert(tk.END,f"===== 考勤汇总 =====\n")
    if attend_start_time:
        txt_output.insert(tk.END,f"开始时间:{attend_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    if attend_end_time:
        dur = (attend_end_time - attend_start_time).seconds
        txt_output.insert(tk.END,f"结束时间:{attend_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        txt_output.insert(tk.END,f"考勤时长:{dur} 秒\n")
    txt_output.insert(tk.END,f"应到总人数:{expected} 人\n")
    txt_output.insert(tk.END,f"累计到场总人数:{len(all_arrive_set)} 人\n")
    txt_output.insert(tk.END,f"已签到总人数:{all_sign_num} 人\n")
    txt_output.insert(tk.END,f"在册签到出勤率:{sign_rate:.1f}%\n")
    txt_output.insert(tk.END,f"未签到人数:{absent_count} 人\n\n")

    txt_output.insert(tk.END,f"【已签到名单】\n")
    for name in sorted(all_sign_set):
        txt_output.insert(tk.END,f"· {name}\n")
    txt_output.insert(tk.END,f"\n【在册缺勤名单】\n")
    if absent_list:
        for name in sorted(absent_list):
            txt_output.insert(tk.END,f"· {name}\n")
    else:
        txt_output.insert(tk.END,f"全员到齐\n")

def export_attendance_log():
    if not all_sign_set and not attend_start_time:
        messagebox.showwarning("提示","暂无考勤数据可导出")
        return
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name = f"课堂考勤日志_{now}.txt"
    with open(save_name,"w",encoding="utf-8") as f:
        f.write(txt_output.get(1.0,tk.END))
    messagebox.showinfo("导出成功",f"日志已保存至：{save_name}")

def select_image():
    global img_path
    img_path=filedialog.askopenfilename(filetypes=[("图片","*.jpg *.png")])
    lbl_path.config(text=img_path)

def image_attendance():
    global img_path
    if not img_path:
        messagebox.showwarning("警告","先选图片")
        return
    img=cv2.imread(img_path)
    frame_out,detected,expected,rate,front,middle,back=analysis_frame(img,rotate_90=False)
   
    show_image_to_gui(frame_out)
    update_log_text(detected,expected,rate,front,middle,back)

def camera_loop():
    global last_frame,last_result,last_save_time
    if not camera_running:
        return
    use_frame=None
    with process_lock:
        if last_frame is not None:
            use_frame=last_frame
            last_frame=None
    if use_frame is not None:
        frame_out,detected,expected,rate,front,middle,back=analysis_frame(use_frame)
        global max_detected_count
        if detected>max_detected_count:max_detected_count=detected
        #摄像展示区域
        #show_image_to_gui(frame_out)
        update_log_text(detected,expected,rate,front,middle,back)
        last_result=(detected,expected,rate,front,middle,back)
    now=time.time()
    if now-last_save_time>=SAVE_INTERVAL and last_result is not None:
        detected,expected,rate,front,middle,back=last_result
        with open("实时快照.txt","a",encoding="utf-8") as f:
            f.write(f"时间:{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"现场:{detected} 签到:{len(all_sign_set)}\n")
        last_save_time=now
    root.after(30,camera_loop)

def stop_camera():
    global cap,camera_running,attend_end_time
    camera_running=False
    all_arrive_set.clear()#开始考勤时清空累计集合
    attend_end_time = datetime.now()
    time.sleep(0.08)
    with process_lock:last_frame=None
    if cap:cap.release();cap=None
    btn_camera_start.config(state=tk.NORMAL)
    btn_camera_stop.config(state=tk.DISABLED)
    print(" 考勤已结束")

def start_camera():
    global cap,camera_running,max_detected_count,attend_start_time,cam_sign_set
    cam_sign_set.clear()
    if camera_running:return
    max_detected_count=0
    attend_start_time = datetime.now()
  
    cam_url="http://admin:admin@10.67.134.123:8081/video"
    cap=cv2.VideoCapture(cam_url,cv2.CAP_FFMPEG)
    if not cap.isOpened():
        messagebox.showerror("错误","打不开摄像头")
        return
    camera_running=True
    btn_camera_start.config(state=tk.DISABLED)
    btn_camera_stop.config(state=tk.NORMAL)
    threading.Thread(target=stream_read_thread,daemon=True).start()
    camera_loop()

# 二维码签到自动刷新循环
def refresh_loop():
    global need_refresh
    if need_refresh:
        update_log_text(0,0,0.0,0,0,0)
        need_refresh = False
    root.after(500, refresh_loop)

# 绑定按钮
btn_select_img.config(command=select_image)
btn_img_detect.config(command=image_attendance)
btn_camera_start.config(command=start_camera)
btn_camera_stop.config(command=stop_camera)
btn_export_log.config(command=export_attendance_log)

if __name__=="__main__":
    # 关闭守护线程，延长等待时间
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    # 等待足够时长让flask完全启动
    time.sleep(5)
    generate_qr_to_file()
    refresh_loop()  # 启动自动刷新
    root.mainloop()
