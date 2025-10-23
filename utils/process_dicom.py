import cv2
import os
import array
from ultralytics import YOLO
import torch
from torchvision import transforms, models
from PIL import Image
from timm import create_model
import torch.nn as nn

sagittal_classification_arr_models = {
    'pfirrmann-EfficientViT-B2-C_v1': [0, -1, 'C'],
    'pfirrmann-EfficientViT-B2-C_v2': [0, 0, 'C'],
    'pfirrmann-EfficientViT-B2-R_v3': [0, 1, 'R'],
    'pfirrmann-EfficientViT-L2-C_v1': [1, 1, 'C'],
    'pfirrmann-EfficientViT-L2-R_v2': [1, 0, 'R'],
    'pfirrmann-Inception-V3-C_v1': [2, 1, 'C'],
    'pfirrmann-Inception-V3-R_v2': [2, 0, 'R']
}

axial_classification_arr_models = {
    'schizas-EfficientViT-B2-C_v1': [0, 0, 'C'],
    'schizas-EfficientViT-L2-C_v1': [1, 1, 'C'],
    'schizas-EfficientViT-L2-R_v2': [1, 0, 'R'],
    'schizas-Inception-V3-C_v1': [2, 1, 'C'],
    'schizas-Inception-V3-R_v2': [2, 0, 'R']
}

def process_dicom_file(path: str, model: str, arr_models: array):
    output_filename = path.replace('uploads/', 'results/')

    sagittal_detection_model = arr_models[0]
    axial_detection_model = arr_models[1]

    sagittal_model = YOLO(f'models/{sagittal_detection_model}.pt')
    axial_model = YOLO(f'models/{axial_detection_model}.pt')

    results = sagittal_model(path) if model == 'sagittal' else axial_model(path)
    
    boxes = []
    for *xyxy, conf, cls in results[0].boxes.data:
        x1, y1, x2, y2 = map(int, xyxy)
        boxes.append([x1, y1, x2, y2])

    merged_boxes = merge_boxes(boxes, iou_threshold=0.5)
    draw_and_save_result(path, merged_boxes, output_filename)
    return merged_boxes

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def merge_boxes(boxes, iou_threshold=0.8):
    merged_boxes = []
    while boxes:
        current_box = boxes.pop(0)
        overlapping_boxes = []
        for i, box in enumerate(boxes):
            iou = calculate_iou(current_box, box)
            if iou > iou_threshold:
                overlapping_boxes.append(i)
        for index in sorted(overlapping_boxes, reverse=True):
            current_box = [(current_box[0] + boxes[index][0]) // 2,
                            (current_box[1] + boxes[index][1]) // 2,
                            (current_box[2] + boxes[index][2]) // 2,
                            (current_box[3] + boxes[index][3]) // 2]
            boxes.pop(index)
        merged_boxes.append(current_box)
    return merged_boxes

def draw_and_save_result(image_path, boxes, output_filename):
    img = cv2.imread(image_path)
    boxes = sorted(boxes, key=lambda x: x[1])
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        view_img = cv2.imread(image_path)
        for j, box2 in enumerate(boxes):
            x1, y1, x2, y2 = box2
            if i != j:
                cv2.rectangle(view_img, (x1, y1), (x2, y2), (225, 255, 225), 1)
            else:
                cv2.rectangle(view_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        filename = output_filename.split('.')[0]
        ext = output_filename.split('.')[1]
        cv2.imwrite(f"{filename}_view_{i}.{ext}", view_img)
    cv2.imwrite(output_filename, img)

def crop_yolo_dataset(image_path: str, output_dir: str, coordinates: array, padding_percent: int, classification: str, arr_models: array):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    img_file = image_path.split('/')[-1]
    img_height, img_width = image.shape[:2]

    coordinates = sorted(coordinates, key=lambda x: x[1])
    
    predict_result = []
    for i, coor in enumerate(coordinates):
        x1 = coor[0]
        y1 = coor[1]
        x2 = coor[2]
        y2 = coor[3]

        padding_x = int(((x2 - x1) * (padding_percent / 100)) / 2)
        padding_y = int(((y2 - y1) * (padding_percent / 100)) / 2)
        
        x_min = max(0, x1 - padding_x)
        y_min = max(0, y1 - padding_y)
        x_max = min(img_width, x2 + padding_x)
        y_max = min(img_height, y2 + padding_y)
        
        cropped = image[y_min:y_max, x_min:x_max]
        
        base_name = os.path.splitext(img_file)[0]
        
        crop_filename = f"{base_name}_{i}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        
        if cropped.size > 0:
            cv2.imwrite(crop_path, cropped)
            predict_result.append(predict_class(crop_path, classification, arr_models))
    
    return predict_result

BASE_MODEL_NAME = [
    { 'model': 'efficientvit_b2.r256_in1k', 'size': 256 },
    { 'model': 'efficientvit_l2.r384_in1k', 'size': 384 },
    { 'model': 'inceptionv3', 'size': 299 }
]
FC = [32, 64]
grade_pfirrmann = ['I', 'II', 'III', 'IV', 'V']
grade_schizas = ['A', 'B', 'C', 'D']

def predict_class(image_path: str, classification: str, arr_models: array):
    sagittal_classification_model = arr_models[2]
    axial_classification_model = arr_models[3]

    if classification == 'pfirrmann':
        num_classes = 5
        if 'EfficientViT-B2' in sagittal_classification_model:
            model_config = sagittal_classification_arr_models[sagittal_classification_model][0]
            fc_config = sagittal_classification_arr_models[sagittal_classification_model][1]
            mode = sagittal_classification_arr_models[sagittal_classification_model][2]
        elif 'EfficientViT-L2' in sagittal_classification_model:
            model_config = sagittal_classification_arr_models[sagittal_classification_model][0]
            fc_config = sagittal_classification_arr_models[sagittal_classification_model][1]
            mode = sagittal_classification_arr_models[sagittal_classification_model][2]
        else:
            model_config = sagittal_classification_arr_models[sagittal_classification_model][0]
            fc_config = sagittal_classification_arr_models[sagittal_classification_model][1]
            mode = sagittal_classification_arr_models[sagittal_classification_model][2]
        return run_model(
            image_path,
            classification,
            f'models/{sagittal_classification_model}.pth',
            model_config,
            fc_config,
            num_classes,
            mode
        )
    else:
        num_classes = 4
        if 'EfficientViT-B2' in axial_classification_model:
            model_config = axial_classification_arr_models[axial_classification_model][0]
            fc_config = axial_classification_arr_models[axial_classification_model][1]
            mode = axial_classification_arr_models[axial_classification_model][2]
        elif 'EfficientViT-L2' in axial_classification_model:
            model_config = axial_classification_arr_models[axial_classification_model][0]
            fc_config = axial_classification_arr_models[axial_classification_model][1]
            mode = axial_classification_arr_models[axial_classification_model][2]
        else:
            model_config = axial_classification_arr_models[axial_classification_model][0]
            fc_config = axial_classification_arr_models[axial_classification_model][1]
            mode = axial_classification_arr_models[axial_classification_model][2]
        return run_model(
            image_path,
            classification,
            f'models/{axial_classification_model}.pth',
            model_config,
            fc_config,
            num_classes,
            mode
        )

def run_model(
    image_path: str,
    classification: str,
    model_path: str,
    model_config: int,
    fc_config: int,
    num_classes: int,
    mode: str
):
    MODEL_NAME = BASE_MODEL_NAME[model_config]['model']
    SIZE = BASE_MODEL_NAME[model_config]['size']
    CHECKPOINT_PATH = model_path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if MODEL_NAME == 'inceptionv3':
        model = models.inception_v3(weights=None, aux_logits=True, init_weights=True)
    elif fc_config == -1:
        model = create_model(MODEL_NAME, pretrained=False)
    else:
        model = create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)

    if fc_config in [0, 1]:
        base_head = FC[fc_config]
        if MODEL_NAME == 'inceptionv3':
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(in_features, base_head),
                nn.ReLU(),
                nn.Linear(base_head, base_head // 2),
                nn.ReLU(),
                nn.Linear(base_head // 2, base_head // 4),
                nn.ReLU(),
                nn.Linear(base_head // 4, num_classes if mode == 'C' else 1)
            )
        else:
            in_features = model.head.classifier[4].in_features
            model.head.classifier[4] = nn.Sequential(
                nn.Linear(in_features, base_head),
                nn.ReLU(),
                nn.Linear(base_head, base_head // 2),
                nn.ReLU(),
                nn.Linear(base_head // 2, base_head // 4),
                nn.ReLU(),
                nn.Linear(base_head // 4, num_classes if mode == 'C' else 1)
            )

    if MODEL_NAME == 'inceptionv3':
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes if mode == 'C' else 1)
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    else:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE), strict=False)
    
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((SIZE, SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    input_image = Image.open(image_path).convert('RGB')
    if mode == 'C':
        input_tensor = transform(input_image).unsqueeze(0)
        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            _, predicted_class = torch.max(output, 1)

        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class.item()].item()
        confidence = "{:.2f}".format(confidence)

        if classification == 'pfirrmann':
            return f"[{confidence}] Grade {grade_pfirrmann[predicted_class.item()]}"
        else:
            return f"[{confidence}] Grade {grade_schizas[predicted_class.item()]}"
    else:
        input_tensor = transform(input_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
            predicted_value_float = output.item() - 1.0 if MODEL_NAME == 'inceptionv3' else output.item()

        predicted_class_int = int(round(predicted_value_float))

        max_idx = (len(grade_pfirrmann) - 1) if classification == 'pfirrmann' else (len(grade_schizas) - 1)
        
        predicted_class_int = max(0, min(max_idx, predicted_class_int))
        diff = abs(predicted_value_float - predicted_class_int)
        conf_val = max(0.0, 1.0 - (diff / 0.5))
        conf_val = min(1.0, conf_val)
        confidence_str = "{:.2f}".format(conf_val)

        if classification == 'pfirrmann':
            return f"[{confidence_str}] Grade {grade_pfirrmann[predicted_class_int]}"
        else:
            return f"[{confidence_str}] Grade {grade_schizas[predicted_class_int]}"