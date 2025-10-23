from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import zipfile
import shutil
import requests
import pydicom
import numpy as np
from PIL import Image
from utils.process_dicom import process_dicom_file, crop_yolo_dataset
from pymongo import MongoClient
import bcrypt
import jwt
import datetime
from middleware import token_required
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
HOST = 'https://cool-narwhal-genuinely.ngrok-free.app'
ADMIN_PASSWORD_HASHED=b'$2b$12$X411NczuZxdBU1NH35JC2e8Dq7zXq2h.KDAsgJz4RnJ.QvN1JMK3.' # ISTTSxUdayana
SECRET_KEY = 'ISTTSxUdayanaProject'

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['SECRET_KEY'] = SECRET_KEY

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

client = MongoClient('mongodb+srv://admin_sdp:L4G5MQZoL6GK4MSG@cluster0.uod7hz7.mongodb.net/')
db = client['db_ta']
users_collection = db['users']

@app.route('/check_token', methods=['POST'])
def check_token():
    token = request.json.get('token')
    if not token:
        return jsonify({'message': 'Token is required'}), 400
    try:
        decoded = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return jsonify({'message': 'Token is valid', 'user': decoded}), 200
    except jwt.ExpiredSignatureError:
        return jsonify({'message': 'Token has expired'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'message': 'Invalid token'}), 401

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or 'username' not in data or 'password' not in data:
        return jsonify({'message': 'Username and Password must be filled'}), 400

    username = data['username']
    password = data['password']

    if username == 'admin':
        if bcrypt.checkpw(password.encode('utf-8'), ADMIN_PASSWORD_HASHED):
            token = generate_token(username='admin', role='admin', fullname='Administrator')
            return jsonify({'message': 'Login successful', 'role': 'admin', 'token': token}), 200
        else:
            return jsonify({'message': 'Incorrect password'}), 401

    user = users_collection.find_one({'username': username})
    if user:
        if bcrypt.checkpw(password.encode('utf-8'), user['password']):
            token = generate_token(username=username, role='user', fullname=user['fullname'])
            return jsonify({'message': 'Login successful', 'role': 'user', 'token': token}), 200
        else:
            return jsonify({'message': 'Incorrect password'}), 401
    else:
        return jsonify({'message': 'User not found'}), 404

@app.route('/register', methods=['POST'])
@token_required(allowed_roles=['admin'])
def register():
    data = request.get_json()

    if not data or not all(key in data for key in ('fullname', 'username', 'password')):
        return jsonify({'message': 'All fields (fullname, username, password) are required'}), 400

    fullname = data['fullname'].strip()
    username = data['username'].strip()
    password = data['password'].strip()

    if not fullname or not username or not password:
        return jsonify({'message': 'All fields must be non-empty'}), 400

    if username == 'admin':
        return jsonify({'message': 'Cannot register admin'}), 403

    if users_collection.find_one({'username': username}):
        return jsonify({'message': 'Username already exists'}), 409

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

    new_user = {
        'fullname': fullname,
        'username': username,
        'password': hashed_pw,
        'models': 'sagittal-disc-YOLOv11_v4|axial-spinal-cord-YOLOv11_v3|pfirrmann-Inception-V3-C_v1|schizas-EfficientViT-L2-C_v1',
        'geom': [],
        'history': []
    }

    users_collection.insert_one(new_user)

    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/user/delete/<username>', methods=['DELETE'])
@token_required(allowed_roles=['admin'])
def delete_user(username):
    if username == 'admin':
        return jsonify({'message': 'Cannot delete admin user'}), 403

    user = users_collection.find_one({'username': username})
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    for history in user['history']:
        shutil.rmtree(os.path.join(app.config['UPLOAD_FOLDER'], history['id']))
        shutil.rmtree(os.path.join(app.config['RESULT_FOLDER'], history['id']))

    users_collection.delete_one({ 'username': username })

    return jsonify({'message': 'User deleted successfully'}), 200

@app.route('/user/all', methods=['GET'])
@token_required(allowed_roles=['admin'])
def get_all_user():
    users = users_collection.find({}, {'_id': 0, 'password': 0})
    user_list = []
    for user in users:
        user_list.append(user)
    return jsonify(user_list), 200

@app.route('/upload', methods=['POST'])
@token_required(allowed_roles=['user'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    model1 = request.form.get('model1', '').strip()
    model2 = request.form.get('model2', '').strip()
    model3 = request.form.get('model3', '').strip()
    model4 = request.form.get('model4', '').strip()
    models = f'{model1}|{model2}|{model3}|{model4}'

    filename = secure_filename(file.filename)
    id = str(uuid.uuid4())
    base_dir = app.config['UPLOAD_FOLDER']
    os.makedirs(base_dir, exist_ok=True)

    if filename.endswith('.zip'):
        zip_path = os.path.join(base_dir, filename)
        file.save(zip_path)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        os.rename(os.path.join(base_dir, filename[:-4]), os.path.join(base_dir, id))
        os.remove(zip_path)

        dicom_path = os.path.join(base_dir, id)
        os.makedirs(os.path.join(dicom_path, 'sagittal'), exist_ok=True)
        os.makedirs(os.path.join(dicom_path, 'axial'), exist_ok=True)

        geom_sag = {}
        geom_axi = {}
        sag_count = 1
        axi_count = 1

        for file_name in sorted(os.listdir(dicom_path)):
            if os.path.isdir(os.path.join(dicom_path, file_name)):
                continue
            
            dicom_image = pydicom.dcmread(os.path.join(dicom_path, file_name), force=True)

            desc = dicom_image.get("SeriesDescription", "").lower()
            modality = dicom_image.get("Modality", "")
            is_t2_sag = "t2_tse_sag" in desc
            is_t2_axi = "t2_tse_tra" in desc

            if modality == "MR" and (is_t2_sag or is_t2_axi):
                image_array = dicom_image.pixel_array

                image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array)) * 255
                image_array = image_array.astype(np.uint8)

                if is_t2_sag:
                    filename_result = f'sagittal/{file_name}_{str(sag_count).zfill(3)}.jpg'
                    sag_count += 1
                else:
                    filename_result = f'axial/{file_name}_{str(axi_count).zfill(3)}.jpg'
                    axi_count += 1

                output_path = os.path.join(dicom_path, filename_result)
                image = Image.fromarray(image_array)
                image.save(output_path)

                geom = get_geometry_info(dicom_image)
                if not geom["valid"]:
                    return jsonify({'message': 'Invalid dicom file info'}), 400

                geom = sanitize_geom(geom)

                if is_t2_sag:
                    geom_sag[file_name] = geom
                else:
                    geom_axi[file_name] = geom
            
            os.remove(os.path.join(dicom_path, file_name))

        users_collection.update_one(
            { 'username': request.user['username'] },
            {
                '$set': {
                    'models': models,
                    'geom': {
                        'sagittal': geom_sag,
                        'axial': geom_axi
                    }
                }
            }
        )
    else:
        return jsonify({'message': 'File must be .zip'}), 400

    return jsonify({
        'message': 'File uploaded successfully',
        'id': id,
        'series': ['-']
    }), 200

@app.route('/get-sagittal-axial/<id>/<series>', methods=['GET'])
@token_required(allowed_roles=['user'])
def get_sagittal_axial(id, series):
    base_dir = app.config['UPLOAD_FOLDER']
    dicom_path = os.path.join(base_dir, id)
    if series != '-':
        dicom_path = os.path.join(base_dir, id, series)

    if not os.path.exists(dicom_path):
        return jsonify({'message': 'Invalid path'}), 400

    sag_prefix = os.path.join(dicom_path, 'sagittal')
    axi_prefix = os.path.join(dicom_path, 'axial')

    sag_path = [f'{HOST}/images/{id}/{series}/sagittal/{sag}/uploads' for sag in os.listdir(sag_prefix)]
    axi_path = [f'{HOST}/images/{id}/{series}/axial/{axi}/uploads' for axi in os.listdir(axi_prefix)]

    return jsonify({
        'sagittal_url': sag_path,
        'axial_url': axi_path
    })

@app.route('/images/<id>/<series>/<view>/<filename>/<folder>', methods=['GET'])
def get_image(id, series, view, filename, folder):
    file_path = os.path.join(folder, id, view, filename)
    if series != '-':
        file_path = os.path.join(folder, id, series, view, filename)

    if not os.path.exists(file_path):
        return jsonify({'message': "File not found"}), 404
    
    response = make_response(send_file(file_path))
    response.headers['ngrok-skip-browser-warning'] = 'true'
    
    return response

@app.route('/results/sagittal/<id>/<series>', methods=['POST'])
@token_required(allowed_roles=['user'])
def process_result_sagittal(id, series):
    if 'sagittal' not in request.json:
        return jsonify({'message': 'Invalid request body'}), 400
    if not isinstance(request.json['sagittal'], int):
        return jsonify({'message': 'Field sagittal must be a number'}), 400
    
    username = request.user['username']
    user = users_collection.find_one({ 'username': username })
    models = user['models'].split('|')

    base_dir = app.config['RESULT_FOLDER']
    os.makedirs(base_dir, exist_ok=True)

    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id)
    result_dir = os.path.join(base_dir, id)
    if series != '-':
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id, series)
        result_dir = os.path.join(base_dir, id, series)
    os.makedirs(result_dir, exist_ok=True)
    
    os.makedirs(os.path.join(result_dir, 'sagittal'), exist_ok=True)

    sagittal = get_filename_paths(upload_dir, 'sagittal', str(request.json['sagittal']).zfill(3) + '.jpg')

    boxes = process_dicom_file(sagittal, 'sagittal', models)
    class_result = crop_yolo_dataset(sagittal, os.path.join(result_dir, 'sagittal'), boxes, 30, 'pfirrmann', models)
    
    result_path = sagittal.replace('uploads/', 'results/')
    axi_total_count = draw_line(result_path, user)
    
    sag_filename = sagittal.split('/')[-1]
    result_sagittal = {
        'result': f'{HOST}/images/{id}/{series}/sagittal/{sag_filename}/results',
        'view': [
            f'{HOST}/images/{id}/{series}/sagittal/{sag_filename.split(".")[0]}_view_{i}.jpg/results'
            for i in range(len(class_result))
        ],
        'line': [
            f'{HOST}/images/{id}/{series}/sagittal/{sag_filename.split(".")[0]}_line_{i}.jpg/results'
            for i in axi_total_count
        ],
        'cropped': [
            {
                'url': f'{HOST}/images/{id}/{series}/sagittal/{sag_filename.split(".")[0]}_{i}.jpg/results',
                'result': class_result[i]
            }
            for i in range(len(class_result))
        ]
    }
    saved_result_sagittal = {
        'result': f'images/{id}/{series}/sagittal/{sag_filename}/results',
        'view': [
            f'images/{id}/{series}/sagittal/{sag_filename.split(".")[0]}_view_{i}.jpg/results'
            for i in range(len(class_result))
        ],
        'cropped': [
            {
                'url': f'images/{id}/{series}/sagittal/{sag_filename.split(".")[0]}_{i}.jpg/results',
                'result': class_result[i]
            }
            for i in range(len(class_result))
        ]
    }

    update_result = users_collection.update_one(
        { 'username': username },
        {
            '$push': {
                'history': {
                    'id': id,
                    'data': { 'sagittal': saved_result_sagittal },
                    'created_at': datetime.datetime.now(datetime.timezone.utc)
                }
            }
        }
    )

    if update_result.modified_count == 1:
        return jsonify({
            'sagittal': result_sagittal
        }), 200
    else:
        return jsonify({'message': 'Process error on saving history'}), 404

@app.route('/results/axial/<id>/<series>', methods=['POST'])
@token_required(allowed_roles=['user'])
def process_result_axial(id, series):
    if 'axial' not in request.json:
        return jsonify({'message': 'Invalid request body'}), 400
    if not isinstance(request.json['axial'], list):
        return jsonify({'message': 'Field axial must be an array'}), 400

    username = request.user['username']
    user = users_collection.find_one({ 'username': username })
    models = user['models'].split('|')

    base_dir = app.config['RESULT_FOLDER']
    os.makedirs(base_dir, exist_ok=True)

    upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id)
    result_dir = os.path.join(base_dir, id)
    if series != '-':
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], id, series)
        result_dir = os.path.join(base_dir, id, series)
    os.makedirs(result_dir, exist_ok=True)

    os.makedirs(os.path.join(result_dir, 'axial'), exist_ok=True)

    axial = []
    for ax in request.json['axial']:
        path = get_filename_paths(upload_dir, 'axial', str(ax).zfill(3) + '.jpg')
        if path is not None:
            axial.append(path)

    result_axial = []
    saved_result_axial = []
    for ax in axial:
        boxes = process_dicom_file(ax, 'axial', models)
        class_result = crop_yolo_dataset(ax, os.path.join(result_dir, 'axial'), boxes, 30, 'schizas', models)
        axi_filename = ax.split('/')[-1]
        result_axial.append({
            'result': f'{HOST}/images/{id}/{series}/axial/{axi_filename}/results',
            'cropped': [
                {
                    'url': f'{HOST}/images/{id}/{series}/axial/{axi_filename.split(".")[0]}_{i}.jpg/results',
                    'result': class_result[i]
                }
                for i in range(len(class_result))
            ]
        })
        saved_result_axial.append({
            'result': f'images/{id}/{series}/axial/{axi_filename}/results',
            'cropped': [
                {
                    'url': f'images/{id}/{series}/axial/{axi_filename.split(".")[0]}_{i}.jpg/results',
                    'result': class_result[i]
                }
                for i in range(len(class_result))
            ]
        })
    
    user = users_collection.find_one({'username': username}, {'history': 1})
    if not user or 'history' not in user or not user['history']:
        return jsonify({'message': 'No history found for this user'}), 404

    latest_history = max(user['history'], key=lambda h: h['created_at'])
    latest_id = latest_history.get('id')
   
    operative_method = []
    for i, sag in enumerate(latest_history['data']['sagittal']['cropped']):
        sag_grade = sag['result'].split(' Grade ')[-1]
        axi_grade = result_axial[i]['cropped'][0]['result'].split(' Grade ')[-1]
        operative_result = predict_operative_method(sag_grade, axi_grade)
        operative_method.append(operative_result)

    if not latest_id:
        return jsonify({'message': 'Latest history entry missing ID'}), 400

    update_result = users_collection.update_one(
        {
            'username': username,
            'history.id': latest_id
        },
        {
            '$set': {
                'history.$.data.axial': saved_result_axial,
                'history.$.data.method': operative_method
            }
        }
    )

    if update_result.modified_count == 1:
        return jsonify({
            'axial': result_axial,
            'method': operative_method
        }), 200
    else:
        return jsonify({'error': 'Failed to update axial data'}), 400

@app.route('/history', methods=['GET'])
@token_required(allowed_roles=['user'])
def get_user_history():
    username = request.user['username']
    user = users_collection.find_one({'username': username}, {'history': 1})
    history = sorted(user['history'], key=lambda x: x['created_at'], reverse=True)
    return jsonify(history), 200

@app.route('/history/<history_id>', methods=['GET'])
@token_required(allowed_roles=['user'])
def get_user_history_detail(history_id):
    username = request.user['username']

    user = users_collection.find_one({'username': username}, {'history': 1})
    if not user or 'history' not in user or not user['history']:
        return jsonify({'message': 'No history found for this user'}), 404

    matched_history = next((h for h in user['history'] if str(h.get('id')) == history_id), None)
    
    if not matched_history:
        return jsonify({'message': f'History with id {history_id} not found'}), 404

    data = matched_history['data']

    data['sagittal']['result'] = f"{HOST}/{data['sagittal']['result']}"
    data['sagittal']['view'] = [f'{HOST}/{x}' for x in data['sagittal']['view']]
    data['sagittal']['cropped'] = [{'url': f"{HOST}/{x['url']}", 'result': x['result']} for x in data['sagittal']['cropped']]

    data['axial'] = [{
        'result': f"{HOST}/{x['result']}",
        'cropped': [{'url': f"{HOST}/{y['url']}", 'result': y['result']} for y in x['cropped']]
    } for x in data['axial']]

    return jsonify(matched_history['data']), 200

def get_filename_paths(dir, view, search):
    for file in os.listdir(os.path.join(dir, view)):
        if file.endswith(search):
            return os.path.join(dir, view, file)
    return None

def generate_token(username, role, fullname):
    payload = {
        'username': username,
        'role': role,
        'fullname': fullname,
        'exp': datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=720)
    }
    token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    return token

def predict_operative_method(pfirrmann, schizas):
    operative_method = ''
    
    if schizas in ['A', 'B']:
        operative_method = 'Conservative Therapy'
    else:
        operative_method = 'Posterior Decompression Fusion or Microdecompression'
    
    if pfirrmann in ['I', 'II', 'III']:
        operative_method += ' without Distraction while operation process'
    else:
        operative_method += ' with Distraction while operation process'

    return operative_method

def get_geometry_info(ds):
    try:
        required_tags = ['ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'Rows', 'Columns']
        for tag in required_tags:
            if not hasattr(ds, tag):
                return {"valid": False, "error": f"Tag {tag} hilang"}

        if not ds.ImagePositionPatient or len(ds.ImagePositionPatient) != 3:
             return {"valid": False, "error": "ImagePositionPatient tidak valid"}
        if not ds.ImageOrientationPatient or len(ds.ImageOrientationPatient) != 6:
             return {"valid": False, "error": "ImageOrientationPatient tidak valid"}
        if not ds.PixelSpacing or len(ds.PixelSpacing) != 2 or ds.PixelSpacing[0] <= 0 or ds.PixelSpacing[1] <= 0:
             return {"valid": False, "error": "PixelSpacing tidak valid"}


        ipp = np.array(ds.ImagePositionPatient, dtype=float)
        iop = np.array(ds.ImageOrientationPatient, dtype=float)
        ps = np.array(ds.PixelSpacing, dtype=float)
        rows = ds.Rows
        cols = ds.Columns
        iop_row_vec = iop[:3]
        iop_col_vec = iop[3:]
        normal = np.cross(iop_row_vec, iop_col_vec)
        return {
            "ipp": ipp,
            "iop_row": iop_row_vec,
            "iop_col": iop_col_vec,
            "ps": ps,
            "rows": rows,
            "cols": cols,
            "normal": normal,
            "valid": True
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}

def sanitize_geom(geom):
    sanitized = {}
    for k, v in geom.items():
        if isinstance(v, np.ndarray):
            sanitized[k] = v.tolist()
        elif isinstance(v, np.generic):
            sanitized[k] = v.item()
        else:
            sanitized[k] = v
    return sanitized

def desanitize_geom(geom):
    desanitized = {}
    for k, v in geom.items():
        if isinstance(v, list) and all(isinstance(i, (int, float)) for i in v):
            desanitized[k] = np.array(v, dtype=np.float32)
        elif isinstance(v, (int, float)):
            desanitized[k] = np.array(v)
        else:
            desanitized[k] = v
    return desanitized

def project_point_to_slice(point_3d, target_slice_geometry):
    ipp_target = target_slice_geometry["ipp"]
    iop_row_target = target_slice_geometry["iop_row"]
    iop_col_target = target_slice_geometry["iop_col"]
    ps_target = target_slice_geometry["ps"]

    vec_p = point_3d - ipp_target

    dist_along_y = np.dot(vec_p, iop_col_target)
    dist_along_x = np.dot(vec_p, iop_row_target)

    pixel_row = dist_along_y / ps_target[0]
    pixel_col = dist_along_x / ps_target[1]

    if not np.isfinite(pixel_col) or not np.isfinite(pixel_row):
        return None

    return pixel_col, pixel_row

def draw_line(ori_img_path, user):
    key  = ori_img_path.split('/')[-1].replace('.jpg', '').split('_')[0]
    sag_geom = desanitize_geom(user['geom']['sagittal'][key])
    axi_geoms = user['geom']['axial']

    sag_img_array = cv2.imread(ori_img_path, cv2.IMREAD_GRAYSCALE)
    img_height, img_width = sag_img_array.shape

    for i, key in enumerate(sorted(axi_geoms.keys())):
        axi_geom = desanitize_geom(axi_geoms[key])

        ipp_axi = axi_geom["ipp"]
        iop_row_axi = axi_geom["iop_row"]
        iop_col_axi = axi_geom["iop_col"]
        ps_axi = axi_geom["ps"]
        rows_axi = axi_geom["rows"]
        cols_axi = axi_geom["cols"]

        half_width_vec = iop_row_axi * ps_axi[1] * (cols_axi - 1) / 2
        half_height_vec = iop_col_axi * ps_axi[0] * (rows_axi - 1) / 2
        center_axi = ipp_axi + half_width_vec + half_height_vec
        mid_top = center_axi - half_height_vec
        mid_bottom = center_axi + half_height_vec

        p_mt = project_point_to_slice(mid_top, sag_geom)
        p_mb = project_point_to_slice(mid_bottom, sag_geom)

        fig, ax = plt.subplots(figsize=(8, 8 * img_height / img_width))
        ax.imshow(sag_img_array, cmap='gray', aspect='equal')

        if p_mt and p_mb:
            ax.plot([p_mt[0], p_mb[0]], [p_mt[1], p_mb[1]], color='cyan', linestyle='-', linewidth=1.5)

        ax.axis('off')
        ax.set_xlim(0, img_width - 1)
        ax.set_ylim(img_height - 1, 0)

        output_path = ori_img_path.replace('.jpg', f'_line_{str(i).zfill(3)}.jpg')
        fig.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close(fig)

    return len(sorted(axi_geoms.keys()))

@app.route('/', methods=['GET'])
def hello_world():
    return jsonify({ 'message': 'Hello World!' }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)