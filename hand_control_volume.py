import cv2
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import time
import math
import numpy as np
from collections import deque


class HandControlVolume:
    def __init__(self):
        # 初始化mediapipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh

        # 获取电脑音量范围
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.volume.SetMute(0, None)
            self.volume_range = self.volume.GetVolumeRange()
        except Exception as e:
            print(f"音频设备初始化失败: {str(e)}")
            raise

        # 音量平滑控制
        self.volume_history = deque(maxlen=5)
        self.last_volume = 0
        
        # 初始化参数
        self.gesture_threshold = 50  # 手势检测阈值
        self.smooth_factor = 0.5    # 平滑因子
        self.face_mask_color = (0, 0, 0)  # 人脸遮罩颜色
        self.face_mask_alpha = 0.8  # 人脸遮罩透明度

        # 初始化检测器
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def _smooth_volume(self, new_volume):
        """平滑音量变化"""
        self.volume_history.append(new_volume)
        return sum(self.volume_history) / len(self.volume_history)

    def _draw_volume_bar(self, image, rect_height, rect_percent_text):
        """绘制音量控制条"""
        # 音量百分比文本
        cv2.putText(image, f"{math.ceil(rect_percent_text)}%", (10, 350),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        # 音量条外框
        cv2.rectangle(image, (30, 100), (70, 300), (255, 0, 0), 3)
        # 音量条填充
        cv2.rectangle(image, (30, math.ceil(300 - rect_height)), (70, 300), (255, 0, 0), -1)
        
        # 添加音量刻度
        for i in range(0, 5):
            y = 300 - (i * 50)
            cv2.line(image, (25, y), (75, y), (255, 0, 0), 1)
            cv2.putText(image, f"{i * 25}", (80, y + 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    def _process_hand_landmarks(self, hand_landmarks, image, resize_w, resize_h):
        """处理手部关键点"""
        landmark_list = []
        for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
            landmark_list.append([
                landmark_id, finger_axis.x, finger_axis.y,
                finger_axis.z
            ])
        
        if not landmark_list:
            return None, None, None

        # 获取手指关键点
        thumb_tip = landmark_list[4]
        index_tip = landmark_list[8]
        
        # 计算坐标
        thumb_point = (math.ceil(thumb_tip[1] * resize_w), 
                      math.ceil(thumb_tip[2] * resize_h))
        index_point = (math.ceil(index_tip[1] * resize_w), 
                      math.ceil(index_tip[2] * resize_h))
        middle_point = ((thumb_point[0] + index_point[0]) // 2,
                       (thumb_point[1] + index_point[1]) // 2)

        return thumb_point, index_point, middle_point

    def _process_face_detection(self, image):
        """处理人脸检测"""
        # 创建遮罩层
        overlay = image.copy()
        
        # 检测人脸
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if results.detections:
            for detection in results.detections:
                # 获取人脸边界框
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
                w, h = int(bboxC.width * iw), int(bboxC.height * ih)
                
                # 绘制遮罩
                cv2.rectangle(overlay, (x, y), (x + w, y + h), 
                            self.face_mask_color, -1)
        
        # 合并遮罩层和原图
        image = cv2.addWeighted(overlay, self.face_mask_alpha, 
                              image, 1 - self.face_mask_alpha, 0)
        return image

    def _process_frame(self, image, hands):
        """处理每一帧图像"""
        rect_height = 0
        rect_percent_text = 0
        
        # 处理图像
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        
        # 处理人脸检测
        image = self._process_face_detection(image)
        
        # 处理手势识别
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # 处理手部关键点
                thumb_point, index_point, middle_point = self._process_hand_landmarks(
                    hand_landmarks, image, image.shape[1], image.shape[0])
                
                if all((thumb_point, index_point, middle_point)):
                    # 绘制关键点和连线
                    cv2.circle(image, thumb_point, 10, (255, 0, 255), -1)
                    cv2.circle(image, index_point, 10, (255, 0, 255), -1)
                    cv2.circle(image, middle_point, 10, (255, 0, 255), -1)
                    cv2.line(image, thumb_point, index_point, (255, 0, 255), 5)

                    # 计算手指距离
                    line_len = math.hypot((index_point[0] - thumb_point[0]),
                                        (index_point[1] - thumb_point[1]))

                    # 音量映射和平滑处理
                    vol = np.interp(line_len, [50, 300], [self.volume_range[0], self.volume_range[1]])
                    vol = self._smooth_volume(vol)
                    
                    # 更新显示数据
                    rect_height = np.interp(line_len, [50, 300], [0, 200])
                    rect_percent_text = np.interp(line_len, [50, 300], [0, 100])

                    # 设置系统音量
                    try:
                        self.volume.SetMasterVolumeLevel(vol, None)
                    except Exception as e:
                        print(f"设置音量失败: {str(e)}")

        return image, rect_height, rect_percent_text

    def recognize(self):
        """主函数"""
        fpsTime = time.time()

        try:
            # OpenCV读取视频流
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise Exception("无法打开摄像头")

            # 视频分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

            with self.mp_hands.Hands(min_detection_confidence=0.7,
                                   min_tracking_confidence=0.5,
                                   max_num_hands=2) as hands:
                while cap.isOpened():
                    success, image = cap.read()
                    if not success:
                        print("无法获取摄像头画面")
                        continue

                    try:
                        # 处理帧
                        image, rect_height, rect_percent_text = self._process_frame(image, hands)
                        
                        # 绘制音量控制条
                        self._draw_volume_bar(image, rect_height, rect_percent_text)

                        # 显示FPS
                        cTime = time.time()
                        fps_text = 1 / (cTime - fpsTime)
                        fpsTime = cTime
                        # cv2.putText(image, f"FPS: {int(fps_text)}", (10, 70),
                        #           cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

                        # 显示画面
                        cv2.imshow('Hand Gesture Volume Control', image)
                        
                    except Exception as e:
                        print(f"处理帧时发生错误: {str(e)}")
                        continue

                    # 检查退出条件
                    if cv2.waitKey(5) & 0xFF == 27 or cv2.getWindowProperty('Hand Gesture Volume Control', cv2.WND_PROP_VISIBLE) < 1:
                        break

        except Exception as e:
            print(f"程序运行出错: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        control = HandControlVolume()
        control.recognize()
    except Exception as e:
        print(f"程序初始化失败: {str(e)}")
