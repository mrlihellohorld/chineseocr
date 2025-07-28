import os
import numpy as np
import torch
import cv2
try:
    import onnx
    import onnxruntime as ort
    from PIL import Image
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("警告: ONNX相关依赖未安装，ONNX转换功能将不可用")

class SmokingDetector:
    def __init__(self, smoking_conf_threshold=0.7, iou_threshold=0.5, device=0, debug=False):
        """
        初始化抽烟检测器
        """
        current_file_path = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file_path)
        
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        # 1. 初始化抽烟检测模型
        model_id = os.path.join(current_folder, '../models/cv_tinynas_object-detection_damoyolo_cigarette')
        
        # 转换设备格式 (ModelScope使用'gpu:X'或'cpu')
        device_str = f'gpu:{device}' if isinstance(device, int) and device >= 0 else 'cpu'
        self.half_precision = False
        self.device = device
        self.debug = debug
        
        # 决定是否使用ONNX
        self.use_onnx = True
        if self.use_onnx:
            self.img_size = (640, 640)  # ONNX模型输入尺寸
            onnx_path = '/workspace/JZ_VideoBehaviorAnalysis/models/onnx/smoking_detection_model.onnx'
            
            if os.path.exists(onnx_path):
                self._init_onnx_model(onnx_path)
            else:
                print(f"⚠️ ONNX模型不存在: {onnx_path}")
                print("回退到ModelScope模型...")
                self.use_onnx = False
        
        if not self.use_onnx:
            # 创建抽烟检测pipeline
            self.cigarette_detection = pipeline(
                Tasks.domain_specific_object_detection,
                model=model_id,
                device=device_str,
                trust_remote_code=True
            )
            
        self.smoking_conf_threshold = smoking_conf_threshold
        self.iou_threshold = iou_threshold
        self.debug = debug

    def _init_onnx_model(self, onnx_path):
        """初始化ONNX模型推理会话"""
        print("=== 初始化ONNX模型 ===")
        
        # 配置执行提供者
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        
        # 创建ONNX会话
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.ort_session = ort.InferenceSession(
            onnx_path, 
            sess_options=session_options,
            providers=providers
        )
        
        # 获取输入输出信息
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        
        print(f"ONNX模型加载成功, 使用执行提供者: {self.ort_session.get_providers()}")
        print(f"输入名称: {self.input_name}")
        print(f"输出名称: {self.output_names}")

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（交并比）
        """
        # 计算交集的坐标
        x_inter_min = max(box1[0], box2[0])
        y_inter_min = max(box1[1], box2[1])
        x_inter_max = min(box1[2], box2[2])
        y_inter_max = min(box1[3], box2[3])
        
        # 计算交集面积
        intersection_area = max(0, x_inter_max - x_inter_min) * max(0, y_inter_max - y_inter_min)
        
        # 计算两个边界框的面积
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        # 计算并集面积
        union_area = box1_area + box2_area - intersection_area
        
        # 计算IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou

    def _preprocess_image(self, img):
        """预处理图像为模型输入格式"""
        # 转换为RGB
        if isinstance(img, Image.Image):
            img = img.convert('RGB')
            orig_w, orig_h = img.size
            img = np.array(img)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img.shape[:2]
        
        # 调整大小并归一化
        img_resized = cv2.resize(img, self.img_size)
        img_array = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = img_array[np.newaxis, :]  # 添加batch维度
        
        return img_tensor, (orig_w, orig_h)

    def _postprocess_onnx_output(self, outputs, orig_size):
        """
        后处理ONNX输出为检测结果
        修正版本 - 适配DAMOYOLO的输出格式
        """
        orig_w, orig_h = orig_size
        
        # 添加调试信息
        if self.debug:
            print("=== ONNX输出调试信息 ===")
            print(f"输出数量: {len(outputs)}")
            for i, output in enumerate(outputs):
                print(f"输出[{i}]形状: {output.shape}")
                print(f"输出[{i}]数据类型: {output.dtype}")
                print(f"输出[{i}]数值范围: [{output.min():.4f}, {output.max():.4f}]")
            print("========================")
        
        # 获取主要输出
        main_output = outputs[0]
        
        boxes = []
        scores = []
        labels = []
        
        # DAMOYOLO的输出格式通常是 [batch, num_detections, num_features]
        # num_features = 4 + 1 + num_classes = 4 + 1 + num_classes
        # 对于香烟检测（单类），通常是 [batch, num_detections, 6] 或 [batch, num_detections, 5]
        
        if len(main_output.shape) == 3:
            batch_size, num_detections, num_features = main_output.shape
            if self.debug:
                print(f"3D输出格式: batch_size={batch_size}, num_detections={num_detections}, features={num_features}")
            
            # 取第一个batch
            detections = main_output[0]  # [num_detections, num_features]
            
        elif len(main_output.shape) == 2:
            num_detections, num_features = main_output.shape
            if self.debug:
                print(f"2D输出格式: num_detections={num_detections}, features={num_features}")
            detections = main_output
        else:
            print(f"未知输出格式: {main_output.shape}")
            return {'boxes': np.array([]), 'scores': np.array([]), 'labels': []}
        
        if self.debug:
            print(f"处理的检测数据形状: {detections.shape}")
            print(f"特征数量: {num_features}")
        
        # 根据DAMOYOLO的输出格式解析
        if num_features >= 5:
            # 检查是否为 [x_center, y_center, width, height, confidence, class_scores...]
            # 或者 [x1, y1, x2, y2, confidence, class_scores...]
            
            for i, det in enumerate(detections):
                if num_features == 5:
                    # [x1, y1, x2, y2, confidence] 或 [x_center, y_center, width, height, confidence]
                    confidence = det[4]
                    class_id = 0  # 单类检测，固定为0
                elif num_features == 6:
                    # [x1, y1, x2, y2, confidence, class_id] 或类似格式
                    confidence = det[4]
                    class_id = int(det[5])
                elif num_features > 6:
                    # [x1, y1, x2, y2, obj_conf, class_conf1, class_conf2, ...]
                    # 或 [x_center, y_center, width, height, obj_conf, class_conf1, ...]
                    obj_conf = det[4]
                    class_confs = det[5:]
                    class_id = np.argmax(class_confs)
                    confidence = obj_conf * class_confs[class_id]  # 组合置信度
                else:
                    continue
                
                if self.debug:
                    print(f"检测{i}: 置信度={confidence:.4f}, 类别ID={class_id}, 阈值={self.smoking_conf_threshold}")
                
                if confidence < self.smoking_conf_threshold:
                    if self.debug:
                        print(f"  -> 置信度过低，跳过")
                    continue
                
                # 解析坐标
                # 检查坐标格式：是否为中心点格式还是左上右下格式
                x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
                
                # 判断坐标格式：如果x1 < x2且y1 < y2且都在合理范围内，可能是左上右下格式
                # 否则可能是中心点+宽高格式
                if x1 > 0 and x1 < x2 and y1 > 0 and y1 < y2 and x2 <= 1 and y2 <= 1:
                    # 归一化的左上右下坐标格式
                    x_min = int(x1 * orig_w)
                    y_min = int(y1 * orig_h)
                    x_max = int(x2 * orig_w)
                    y_max = int(y2 * orig_h)
                else:
                    # 可能是中心点+宽高格式 (x_center, y_center, width, height)
                    x_center, y_center, width, height = x1, y1, x2, y2
                    
                    # 转换为左上右下坐标
                    x_min = int((x_center - width/2) * orig_w)
                    y_min = int((y_center - height/2) * orig_h)
                    x_max = int((x_center + width/2) * orig_w)
                    y_max = int((y_center + height/2) * orig_h)
                
                # 确保坐标在图像范围内
                x_min = max(0, min(x_min, orig_w))
                y_min = max(0, min(y_min, orig_h))
                x_max = max(0, min(x_max, orig_w))
                y_max = max(0, min(y_max, orig_h))
                
                if self.debug:
                    print(f"  -> 坐标: ({x_min}, {y_min}, {x_max}, {y_max})")
                
                # 检查边界框的有效性
                if x_max > x_min and y_max > y_min:
                    boxes.append([x_min, y_min, x_max, y_max])
                    scores.append(confidence)
                    labels.append('cigarette')  # 固定标签
                elif self.debug:
                    print(f"  -> 无效边界框，跳过")
                    
        else:
            print(f"未识别的特征数量: {num_features}")
            
        if self.debug:
            print(f"最终结果: {len(boxes)}个检测框")
            
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'labels': labels
        }

    def _detect_smoking(self, img, bboxes):
        """
        检测图片中的抽烟行为
        
        :param img: 输入图片（PIL Image 对象）
        :param bboxes: 人体检测框列表 [[x_min, y_min, x_max, y_max], ...]
        :return: 抽烟检测结果列表，抽烟边界框列表
        """
        if self.use_onnx:
            # ONNX推理流程
            img_tensor, orig_size = self._preprocess_image(img)
            
            # 执行推理
            ort_inputs = {self.input_name: img_tensor}
            ort_outputs = self.ort_session.run(self.output_names, ort_inputs)
            
            # 后处理结果
            result = self._postprocess_onnx_output(ort_outputs, orig_size)
        else:
            # 使用ModelScope模型进行推理
            result = self.cigarette_detection(img)
            
        if self.debug:
            print("检测结果:", result)
            
        smoking_detections = []
        smoke_bboxs = []
        
        # 解析检测结果
        if 'boxes' in result and len(result['boxes']) > 0:
            for i in range(len(result['boxes'])):
                label = result['labels'][i]
                conf = result['scores'][i]
                bbox = result['boxes'][i].tolist() if hasattr(result['boxes'][i], 'tolist') else result['boxes'][i]
                
                if self.debug:
                    print('label', label, 'conf', conf, 'bbox', bbox)
                    
                # 只处理香烟检测结果
                if label == 'cigarette' and conf > self.smoking_conf_threshold:
                    # 检查是否与任何人脸框有交集
                    for person_box in bboxes:
                        iou = self._calculate_iou(person_box, bbox)
                        if iou > 0:
                            smoking_detections.append(True)
                            # 添加边界框和置信度 [x_min, y_min, x_max, y_max, conf]
                            smoke_bboxs.append(bbox + [conf])
                            break  # 找到一个匹配就跳出
        
        return smoking_detections, smoke_bboxs

# 使用示例（如果作为独立脚本运行）
if __name__ == "__main__":
    import cv2
    import time
    from PIL import ImageFont, ImageDraw, Image
    
    detector = SmokingDetector(
        smoking_conf_threshold=0.6,
        iou_threshold=0.5,
        device=6,
        debug=True
    )
    print("检测器初始化完成")
    
    # 测试代码...