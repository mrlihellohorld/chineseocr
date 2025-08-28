"""
@author: lzq
@date: 2025-08-28
@description: 抽烟检测视频评估器，包含正负样本

统计精确率时，按照视频事件触发的方式计算。召回率计算方式基本不变，需要保持和之前计算出来的指标相同。
修改精确率计算方式：读取更多视频，/home/jzadmin/grq/code/Co-DETR/eval_lzq_test/benchmark_video的[01_smoke,02_phone]只读取两个文件夹。
                  把原来01_smoke文件夹中每个视频的检测结果缓存起来，
                  并且对其他文件夹下视频文件进行识别，如果其他文件夹下视频中有检测结果，同样缓存起来。即为误检测样例。
                  并且，如果非01_smoke文件夹检测到抽烟，把对应的图片保存下来，并且把时间信息存储到结果json文件。
                  然后基于多个文件夹下视频的检测结果，计算精确率。
注意：如果非01_smoke文件夹，判断是否触发事件，判断单个视频是否检测有结果时，采用方式为：设置滑动窗口，20，里面超过5帧才算一次正确识别。

"""
import os
import cv2
import numpy as np
import json
from collections import defaultdict, deque
import shutil
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
#from detection_model_yolo_phone import YOLOXDetector
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

def phone_detection_results(detect_results, cat_id=0, score_thr=0.35, person_bbox=None):
    """
    处理检测结果
    支持两种格式：
    1. DAMO-YOLO格式: {'scores': array, 'labels': list, 'boxes': array(N, 5)}
    2. SmokingDetector格式: [{'class_id': 0, 'class_name': 'cigarette', 'bbox': [...], 'score': 0.96458775}]
    其中boxes格式为 [x1, y1, x2, y2, score]
    """
    # 处理SmokingDetector格式（列表格式）
    if isinstance(detect_results, list):
        results_bbox = []
        for detection in detect_results:
            if isinstance(detection, dict) and 'bbox' in detection and 'score' in detection:
                score = detection['score']
                if score >= score_thr:
                    bbox = detection['bbox']
                    if len(bbox) >= 4:
                        # 格式转换：确保输出格式为 [x1, y1, x2, y2, score]
                        bbox_with_score = bbox[:4] + [score]
                        results_bbox.append(bbox_with_score)
        return results_bbox
    
    # 处理DAMO-YOLO格式（字典格式）
    if isinstance(detect_results, dict):
        if 'boxes' not in detect_results or len(detect_results['boxes']) == 0:
            return []
        
        boxes = detect_results['boxes']
        scores = detect_results['scores']
        labels = detect_results['labels'] if detect_results['labels'] else []
        
        # 如果没有检测结果
        if len(boxes) == 0:
            return []
        
        # 过滤分数阈值
        if len(scores) > 0:
            score_mask = scores > score_thr
            if not any(score_mask):
                return []
            boxes = boxes[score_mask]
            scores = scores[score_mask]
            if len(labels) > 0:
                labels = np.array(labels)[score_mask]
        
        # 格式转换：确保输出格式为 [x1, y1, x2, y2, score]
        results_bbox = []
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                if len(box) == 5:
                    # 已经包含分数
                    results_bbox.append(box.tolist())
                else:
                    # 需要添加分数
                    score = scores[i] if i < len(scores) else 0.5
                    bbox_with_score = np.append(box[:4], score)
                    results_bbox.append(bbox_with_score.tolist())
        
        return results_bbox
    
    return []

def time_to_seconds(time_str):
    """
    将时间字符串转换为秒数
    格式: "MM:SS.mmm" 或 "HH:MM:SS.mmm"
    """
    if not time_str or time_str == "":
        return 0
    
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS.mmm
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS.mmm
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    else:
        return 0

def is_time_in_range(frame_time, time_ranges):
    """
    检查帧时间是否在指定的时间范围内
    :param frame_time: 帧时间（秒）
    :param time_ranges: 时间范围列表 [[start_time, end_time], ...]
    :return: 是否在范围内
    """
    for start_str, end_str in time_ranges:
        start_time = time_to_seconds(start_str)
        end_time = time_to_seconds(end_str)
        
        if start_time <= frame_time <= end_time:
            return True
    return False

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class SlidingWindowDetector:
    """滑动窗口检测器，用于判断视频中是否触发事件"""
    
    def __init__(self, window_size=20, threshold=5):
        """
        初始化滑动窗口检测器
        :param window_size: 窗口大小（帧数）
        :param threshold: 阈值（窗口内检测帧数超过此值才认为触发事件）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.window = deque(maxlen=window_size)
        self.detection_count = 0
        self.triggered_events = []  # 记录触发的事件
    
    def add_frame(self, frame_idx, has_detection, frame_time=None):
        """
        添加新帧到滑动窗口
        :param frame_idx: 帧索引
        :param has_detection: 是否有检测结果
        :param frame_time: 帧时间
        :return: 是否触发新事件
        """
        # 如果窗口满了，移除最旧的帧
        if len(self.window) == self.window_size:
            old_frame = self.window[0]
            if old_frame['has_detection']:
                self.detection_count -= 1
        
        # 添加新帧
        frame_info = {
            'frame_idx': frame_idx,
            'has_detection': has_detection,
            'frame_time': frame_time
        }
        self.window.append(frame_info)
        
        if has_detection:
            self.detection_count += 1
        
        # 检查是否触发事件
        if self.detection_count > self.threshold:
            # 检查是否是新事件（避免重复触发）
            if not self.triggered_events or frame_idx - self.triggered_events[-1]['end_frame'] > self.window_size:
                event = {
                    'start_frame': self.window[0]['frame_idx'],
                    'end_frame': frame_idx,
                    'start_time': self.window[0]['frame_time'],
                    'end_time': frame_time,
                    'detection_count': self.detection_count
                }
                self.triggered_events.append(event)
                return True
        
        return False
    
    def get_triggered_events(self):
        """获取所有触发的事件"""
        return self.triggered_events
    
    def has_any_event(self):
        """判断是否有任何事件被触发"""
        return len(self.triggered_events) > 0

class SmokingDetectionVideoEvaluator:
    def __init__(self, base_video_dir, output_dir, device=0):
        """
        初始化视频评估器
        :param base_video_dir: 基础视频目录（包含01_smoke, 02_phone等子文件夹）
        :param output_dir: 输出目录
        :param device: 设备ID
        """
        self.base_video_dir = base_video_dir
        self.output_dir = output_dir
        self.device = device
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "false_positives"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "false_negatives"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "true_positives"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "detection_images"), exist_ok=True)
        
        # 目标检测模型
        config_file = '/home/jzadmin/grq/code/Co-DETR/projects/configs/co_deformable_detr/co_deformable_detr_swin_large_cigarette_finetune.py'
        checkpoint_file = '/home/jzadmin/grq/code/Co-DETR/work_dirs/cigarette_detection/best_bbox_mAP_epoch_30.pth'
        from eval_smoking_detection import SmokingDetector
        self.object_detector = SmokingDetector(config_file, checkpoint_file, device=f'cuda:{device}')
        
        # 存储检测结果缓存
        self.detection_cache = {}
        
        # 存储评估结果
        self.results = {
            'positive_videos': {},  # 01_smoke文件夹的视频结果
            'negative_videos': {},  # 其他文件夹的视频结果
            'metrics': {},
            'false_positive_details': []  # 误检测详细信息
        }
    
    def load_config(self, config_file):
        """
        加载配置文件
        :param config_file: 配置文件路径
        :return: 配置数据
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败 {config_file}: {e}")
            return None
    
    def save_detection_image(self, frame, detected_boxes, video_name, frame_count, frame_time, folder_name):
        """
        保存检测结果图片
        :param frame: 视频帧
        :param detected_boxes: 检测框
        :param video_name: 视频名称
        :param frame_count: 帧数
        :param frame_time: 帧时间
        :param folder_name: 文件夹名称
        """
        try:
            # 创建输出文件夹
            output_folder = os.path.join(self.output_dir, "detection_images", folder_name)
            os.makedirs(output_folder, exist_ok=True)
            
            # 在图片上绘制检测框
            frame_copy = frame.copy()
            for box in detected_boxes:
                x1, y1, x2, y2, score = box
                # 绘制矩形框
                cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # 添加分数标签
                label = f'smoke: {score:.2f}'
                cv2.putText(frame_copy, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 保存图片
            image_name = f"{video_name}_frame_{frame_count}_time_{frame_time:.2f}s.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame_copy)
            
            return image_path
        except Exception as e:
            print(f"保存检测图片失败: {e}")
            return None
    
    def evaluate_positive_video(self, video_path, config_path, debug_mode=False):
        """
        评估正样本视频（01_smoke文件夹）
        :param video_path: 视频路径
        :param config_path: 配置文件路径
        :param debug_mode: 是否开启调试模式
        :return: 评估结果
        """
        # 加载配置文件
        config = self.load_config(config_path)
        if config is None:
            print(f"无法加载配置文件: {config_path}")
            return None
        
        # 获取抽烟行为的时间范围
        smoke_time_ranges = config.get('behaviors', {}).get('抽烟', [])
        if not smoke_time_ranges or smoke_time_ranges[0][0] == "":
            print(f"配置文件中没有有效的抽烟行为时间范围: {config_path}")
            return None
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"正样本视频信息: {os.path.basename(video_path)}")
        print(f"  FPS: {fps:.2f}")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {duration:.2f}秒")
        print(f"  抽烟行为时间范围: {smoke_time_ranges}")
        
        # 初始化结果
        video_result = {
            'video_path': video_path,
            'config_path': config_path,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'smoke_time_ranges': smoke_time_ranges,
            'detected_frames': [],  # 检测到抽烟的帧
            'correct_detections': 0,  # 正确检测的帧数
            'total_detections': 0,    # 总检测帧数
            'has_detection': False,   # 是否有任何检测
            'has_correct_detection': False,  # 是否有正确检测
            'is_positive_sample': True  # 标记为正样本
        }
        
        frame_count = 0
        detected_frames_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_time = frame_count / fps  # 当前帧的时间（秒）
            
            # 每3帧处理一次（提高效率）
            if frame_count % 3 != 0:
                continue
            
            # 抽烟检测
            detect_results = self.object_detector.detect(frame, conf_threshold=0.5)
            
            detected_boxes = phone_detection_results(
                detect_results, cat_id=0, score_thr=0.5, person_bbox=None
            )
            
            # 如果检测到抽烟
            if detected_boxes:
                video_result['total_detections'] += 1
                video_result['has_detection'] = True
                
                # 检查是否在正确的时间范围内
                if is_time_in_range(frame_time, smoke_time_ranges):
                    video_result['correct_detections'] += 1
                    video_result['has_correct_detection'] = True
                    detected_frames_count += 1
                    
                    # 保存检测信息
                    video_result['detected_frames'].append({
                        'frame_count': frame_count,
                        'frame_time': frame_time,
                        'detected_boxes': detected_boxes,
                        'is_correct': True
                    })
                    
                    if debug_mode:
                        print(f"  帧 {frame_count} (时间: {frame_time:.2f}s): 正确检测到抽烟")
                else:
                    # 错误检测（不在时间范围内）
                    video_result['detected_frames'].append({
                        'frame_count': frame_count,
                        'frame_time': frame_time,
                        'detected_boxes': detected_boxes,
                        'is_correct': False
                    })
                    
                    if debug_mode:
                        print(f"  帧 {frame_count} (时间: {frame_time:.2f}s): 错误检测到抽烟")
        
        cap.release()
        
        # 计算视频级别的指标
        precision = video_result['correct_detections'] / video_result['total_detections'] if video_result['total_detections'] > 0 else 0
        recall = 1.0 if video_result['has_correct_detection'] else 0.0  # 只要有一个正确检测就算召回
        
        video_result['precision'] = precision
        video_result['recall'] = recall
        video_result['f1_score'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  检测结果: 总检测帧数={video_result['total_detections']}, 正确检测帧数={video_result['correct_detections']}")
        print(f"  精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {video_result['f1_score']:.4f}")
        
        return video_result
    
    def evaluate_negative_video(self, video_path, folder_name, debug_mode=False):
        """
        评估负样本视频（非01_smoke文件夹）
        :param video_path: 视频路径
        :param folder_name: 文件夹名称
        :param debug_mode: 是否开启调试模式
        :return: 评估结果
        """
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
            return None
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"负样本视频信息: {video_name} (文件夹: {folder_name})")
        print(f"  FPS: {fps:.2f}")
        print(f"  总帧数: {total_frames}")
        print(f"  时长: {duration:.2f}秒")
        
        # 初始化滑动窗口检测器
        sliding_detector = SlidingWindowDetector(window_size=20, threshold=5)
        
        # 初始化结果
        video_result = {
            'video_path': video_path,
            'folder_name': folder_name,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'detected_frames': [],  # 检测到抽烟的帧
            'total_detections': 0,    # 总检测帧数
            'has_detection': False,   # 是否有任何检测
            'triggered_events': [],   # 触发的事件
            'has_event': False,       # 是否触发事件（基于滑动窗口）
            'is_positive_sample': False,  # 标记为负样本
            'false_positive_images': []   # 误检测图片路径
        }
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_time = frame_count / fps  # 当前帧的时间（秒）
            
            # 每3帧处理一次（提高效率）
            if frame_count % 3 != 0:
                continue
            
            # 抽烟检测
            detect_results = self.object_detector.detect(frame, conf_threshold=0.5)
            
            detected_boxes = phone_detection_results(
                detect_results, cat_id=0, score_thr=0.5, person_bbox=None
            )
            
            has_detection = len(detected_boxes) > 0
            
            # 添加到滑动窗口
            is_new_event = sliding_detector.add_frame(frame_count, has_detection, frame_time)
            
            # 如果检测到抽烟
            if detected_boxes:
                video_result['total_detections'] += 1
                video_result['has_detection'] = True
                
                # 保存检测信息
                detection_info = {
                    'frame_count': frame_count,
                    'frame_time': frame_time,
                    'detected_boxes': detected_boxes,
                    'is_false_positive': True  # 负样本中的检测都是误检测
                }
                video_result['detected_frames'].append(detection_info)
                
                # 保存误检测图片
                image_path = self.save_detection_image(frame, detected_boxes, video_name, frame_count, frame_time, folder_name)
                if image_path:
                    video_result['false_positive_images'].append({
                        'image_path': image_path,
                        'frame_count': frame_count,
                        'frame_time': frame_time,
                        'detected_boxes': detected_boxes
                    })
                
                if debug_mode:
                    print(f"  帧 {frame_count} (时间: {frame_time:.2f}s): 误检测到抽烟")
            
            # 如果触发新事件
            if is_new_event:
                print(f"  触发新事件: 帧 {frame_count} (时间: {frame_time:.2f}s)")
        
        cap.release()
        
        # 获取所有触发的事件
        video_result['triggered_events'] = sliding_detector.get_triggered_events()
        video_result['has_event'] = sliding_detector.has_any_event()
        
        print(f"  检测结果: 总检测帧数={video_result['total_detections']}")
        print(f"  触发事件数: {len(video_result['triggered_events'])}")
        print(f"  是否有事件: {video_result['has_event']}")
        
        return video_result
    
    def evaluate_all_videos(self):
        """
        评估所有视频
        """
        print("开始评估检测模型（增强版 - 多文件夹模式）...")
        
        # 定义需要处理的文件夹
        target_folders = ['01_smoke', '02_phone']
        
        # 统计变量
        total_positive_videos = 0
        total_negative_videos = 0
        positive_videos_with_correct_detection = 0
        negative_videos_with_false_detection = 0
        
        # 处理每个文件夹
        for folder_name in target_folders:
            folder_path = os.path.join(self.base_video_dir, folder_name)
            
            if not os.path.exists(folder_path):
                print(f"文件夹不存在: {folder_path}")
                continue
            
            print(f"\n=== 处理文件夹: {folder_name} ===")
            
            # 获取文件夹中的所有视频文件
            video_files = []
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                video_files.extend([f for f in os.listdir(folder_path) if f.lower().endswith(ext.lower())])
            
            print(f"找到 {len(video_files)} 个视频文件")
            
            # 处理每个视频
            for i, video_file in enumerate(video_files):
                video_path = os.path.join(folder_path, video_file)
                video_name = os.path.splitext(video_file)[0]
                
                print(f"\n处理视频 {i+1}/{len(video_files)}: {video_file}")
                
                # 调试模式：只对前3个视频打印详细信息
                debug_mode = i < 3
                
                if folder_name == '01_smoke':
                    # 正样本视频（需要配置文件）
                    config_file = video_name + '.json'
                    config_path = os.path.join(folder_path, config_file)
                    
                    if not os.path.exists(config_path):
                        print(f"  未找到配置文件: {config_file}")
                        continue
                    
                    result = self.evaluate_positive_video(video_path, config_path, debug_mode)
                    if result:
                        self.results['positive_videos'][video_name] = result
                        total_positive_videos += 1
                        
                        if result['has_correct_detection']:
                            positive_videos_with_correct_detection += 1
                
                else:
                    # 负样本视频（不需要配置文件）
                    result = self.evaluate_negative_video(video_path, folder_name, debug_mode)
                    if result:
                        self.results['negative_videos'][video_name] = result
                        total_negative_videos += 1
                        
                        if result['has_event']:
                            negative_videos_with_false_detection += 1
                            
                            # 保存误检测详细信息
                            for event in result['triggered_events']:
                                false_positive_detail = {
                                    'video_name': video_name,
                                    'folder_name': folder_name,
                                    'video_path': video_path,
                                    'event': event,
                                    'false_positive_images': result['false_positive_images']
                                }
                                self.results['false_positive_details'].append(false_positive_detail)
        
        # 计算总体指标
        # 精确率 = 正确检测的正样本视频数 / (正确检测的正样本视频数 + 误检测的负样本视频数)
        true_positives = positive_videos_with_correct_detection
        false_positives = negative_videos_with_false_detection
        false_negatives = total_positive_videos - positive_videos_with_correct_detection
        true_negatives = total_negative_videos - negative_videos_with_false_detection
        
        overall_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        overall_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        # 计算准确率
        overall_accuracy = (true_positives + true_negatives) / (true_positives + false_positives + false_negatives + true_negatives) if (true_positives + false_positives + false_negatives + true_negatives) > 0 else 0
        
        self.results['metrics'] = {
            'total_positive_videos': total_positive_videos,
            'total_negative_videos': total_negative_videos,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1_score': overall_f1,
            'overall_accuracy': overall_accuracy,
            'positive_videos_with_correct_detection': positive_videos_with_correct_detection,
            'negative_videos_with_false_detection': negative_videos_with_false_detection
        }
        
        # 保存结果
        self.save_results()
        self.plot_results()
        
        print(f"\n=== 总体评估结果 ===")
        print(f"正样本视频数: {total_positive_videos}")
        print(f"负样本视频数: {total_negative_videos}")
        print(f"正确检测的正样本视频数: {positive_videos_with_correct_detection}")
        print(f"误检测的负样本视频数: {negative_videos_with_false_detection}")
        print(f"真正例 (TP): {true_positives}")
        print(f"假正例 (FP): {false_positives}")
        print(f"假负例 (FN): {false_negatives}")
        print(f"真负例 (TN): {true_negatives}")
        print(f"总体精确率: {overall_precision:.4f}")
        print(f"总体召回率: {overall_recall:.4f}")
        print(f"总体F1分数: {overall_f1:.4f}")
        print(f"总体准确率: {overall_accuracy:.4f}")
    
    def save_results(self):
        """
        保存评估结果
        """
        results_file = os.path.join(self.output_dir, "enhanced_video_evaluation_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        # 保存详细统计
        stats_file = os.path.join(self.output_dir, "enhanced_video_detailed_stats.txt")
        with open(stats_file, 'w', encoding='utf-8') as f:
            metrics = self.results['metrics']
            
            f.write("抽烟检测视频评估结果（增强版）\n")
            f.write("=" * 50 + "\n")
            f.write(f"正样本视频数: {metrics['total_positive_videos']}\n")
            f.write(f"负样本视频数: {metrics['total_negative_videos']}\n")
            f.write(f"正确检测的正样本视频数: {metrics['positive_videos_with_correct_detection']}\n")
            f.write(f"误检测的负样本视频数: {metrics['negative_videos_with_false_detection']}\n")
            f.write(f"真正例 (TP): {metrics['true_positives']}\n")
            f.write(f"假正例 (FP): {metrics['false_positives']}\n")
            f.write(f"假负例 (FN): {metrics['false_negatives']}\n")
            f.write(f"真负例 (TN): {metrics['true_negatives']}\n")
            f.write(f"总体精确率: {metrics['overall_precision']:.4f}\n")
            f.write(f"总体召回率: {metrics['overall_recall']:.4f}\n")
            f.write(f"总体F1分数: {metrics['overall_f1_score']:.4f}\n")
            f.write(f"总体准确率: {metrics['overall_accuracy']:.4f}\n")
            
            f.write("\n正样本视频详细结果:\n")
            for video_name, result in self.results['positive_videos'].items():
                f.write(f"\n{video_name}:\n")
                f.write(f"  精确率: {result['precision']:.4f}\n")
                f.write(f"  召回率: {result['recall']:.4f}\n")
                f.write(f"  F1分数: {result['f1_score']:.4f}\n")
                f.write(f"  总检测帧数: {result['total_detections']}\n")
                f.write(f"  正确检测帧数: {result['correct_detections']}\n")
                f.write(f"  抽烟行为时间范围: {result['smoke_time_ranges']}\n")
            
            f.write("\n负样本视频详细结果:\n")
            for video_name, result in self.results['negative_videos'].items():
                f.write(f"\n{video_name} (文件夹: {result['folder_name']}):\n")
                f.write(f"  总检测帧数: {result['total_detections']}\n")
                f.write(f"  触发事件数: {len(result['triggered_events'])}\n")
                f.write(f"  是否有事件: {result['has_event']}\n")
                f.write(f"  误检测图片数: {len(result['false_positive_images'])}\n")
        
        # 保存误检测详细信息
        false_positive_file = os.path.join(self.output_dir, "false_positive_details.json")
        with open(false_positive_file, 'w', encoding='utf-8') as f:
            json.dump(self.results['false_positive_details'], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    def plot_results(self):
        """
        绘制结果图表
        """
        metrics = self.results['metrics']
        
        # 创建图表
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 总体指标对比
        labels = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
        values = [metrics['overall_precision'], metrics['overall_recall'], 
                 metrics['overall_f1_score'], metrics['overall_accuracy']]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        bars = ax1.bar(labels, values, color=colors)
        ax1.set_title('Overall Detection Metrics (Enhanced)')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(values):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. 混淆矩阵
        confusion_data = np.array([[metrics['true_positives'], metrics['false_negatives']],
                                  [metrics['false_positives'], metrics['true_negatives']]])
        sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues', ax=ax2,
                   xticklabels=['Predicted Positive', 'Predicted Negative'],
                   yticklabels=['Actual Positive', 'Actual Negative'])
        ax2.set_title('Confusion Matrix')
        
        # 3. 视频检测结果分布
        detection_labels = ['TP Videos', 'FP Videos', 'FN Videos', 'TN Videos']
        detection_values = [
            metrics['true_positives'],
            metrics['false_positives'],
            metrics['false_negatives'],
            metrics['true_negatives']
        ]
        colors = ['#99ff99', '#ff9999', '#ffcc99', '#66b3ff']
        
        ax3.pie(detection_values, labels=detection_labels, colors=colors, autopct='%1.1f%%')
        ax3.set_title('Video Detection Results Distribution')
        
        # 4. 统计信息
        ax4.axis('off')
        stats_text = f"""
        Enhanced Video Evaluation Results
        
        Dataset Statistics:
        - Total Positive Videos: {metrics['total_positive_videos']}
        - Total Negative Videos: {metrics['total_negative_videos']}
        
        Detection Results:
        - True Positives (TP): {metrics['true_positives']}
        - False Positives (FP): {metrics['false_positives']}
        - False Negatives (FN): {metrics['false_negatives']}
        - True Negatives (TN): {metrics['true_negatives']}
        
        Performance Metrics:
        - Precision: {metrics['overall_precision']:.3f}
        - Recall: {metrics['overall_recall']:.3f}
        - F1-Score: {metrics['overall_f1_score']:.3f}
        - Accuracy: {metrics['overall_accuracy']:.3f}
        
        False Positive Events: {len(self.results['false_positive_details'])}
        """
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace')
        ax4.set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'enhanced_video_evaluation_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 设置路径
    base_video_dir = "/home/jzadmin/grq/code/Co-DETR/eval_lzq_test/benchmark_video"
    output_dir = "/home/jzadmin/grq/code/Co-DETR/eval_lzq_test/benchmark_result/enhanced_video_evaluation"
    device = 5  # GPU设备ID
    
    print("开始初始化增强版视频评估器...")
    try:
        # 创建评估器并运行评估
        evaluator = SmokingDetectionVideoEvaluator(base_video_dir, output_dir, device)
        print("增强版视频评估器初始化成功！")
        
        # 运行评估
        print("开始运行增强版视频评估...")
        evaluator.evaluate_all_videos()
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()