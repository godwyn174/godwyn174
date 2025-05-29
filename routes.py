from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np
import pandas as pd
from scikit-learn.ensemble import RandomForestClassifier
from scikit-learn.model_selection import train_test_split
from scikit-learn.preprocessing import LabelEncoder
import cv2
import os
from werkzeug.utils import secure_filename

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.doc', '.docx', '.pdf', '.mp3', '.wav', '.jpg', '.png', '.csv'}

def allowed_file(filename):
    file_ext = os.path.splitext(filename)[1].lower()
    print(f"Checking file: {filename}, extension: {file_ext}")
    return file_ext in ALLOWED_EXTENSIONS

def init_app(app):
    # --- ML Model Setup ---
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(128, activation='relu')(x)
    x = Dense(7, activation='softmax')(x)  # 7 soil types
    soil_model = Model(inputs=base_model.input, outputs=x)

    # Initial crop recommendation dataset
    data = {
        'soil_type': ['Red', 'Alluvial', 'Black', 'Clay', 'Loam', 'Silt', 'Sandy'] * 25,
        'pH': [6.5, 7.0, 6.0, 5.5, 6.2, 6.8, 6.3] * 25,
        'rainfall': [800, 1000, 900, 850, 750, 950, 700] * 25,
        'humidity': [65, 70, 60, 68, 72, 65, 58] * 25,
        'crop': ['Rice', 'Wheat', 'Cotton', 'Maize', 'Soybean', 'Barley', 'Sorghum'] * 25
    }
    df = pd.DataFrame(data)
    soil_encoder = LabelEncoder()
    crop_encoder = LabelEncoder()
    df['soil_type'] = soil_encoder.fit_transform(df['soil_type'])
    df['crop'] = crop_encoder.fit_transform(df['crop'])
    X = df[['soil_type', 'pH', 'rainfall', 'humidity']]
    y = df['crop']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    def extract_rgb(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        rgb = np.mean(img, axis=(0, 1))
        return rgb

    def predict_soil_type(image_path):
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = soil_model.predict(img_array)
        soil_types = ['Red', 'Alluvial', 'Black', 'Clay', 'Loam', 'Silt', 'Sandy']
        return soil_types[np.argmax(prediction)]

    # --- Dataset Generation ---
    def generate_synthetic_soil_data(num_samples=1000):
        soil_types = ['Red', 'Alluvial', 'Black', 'Clay', 'Loam', 'Silt', 'Sandy']
        rgb_ranges = {
            'Red': ([150, 50, 50], [200, 100, 100]),
            'Alluvial': ([100, 80, 60], [150, 120, 100]),
            'Black': ([20, 20, 20], [50, 50, 50]),
            'Clay': ([120, 100, 80], [160, 140, 120]),
            'Loam': ([80, 60, 40], [120, 100, 80]),
            'Silt': ([140, 130, 110], [180, 170, 150]),
            'Sandy': ([200, 180, 140], [240, 220, 180])
        }
        data = []
        for _ in range(num_samples):
            soil_type = np.random.choice(soil_types)
            rgb_min, rgb_max = rgb_ranges[soil_type]
            r = np.random.randint(rgb_min[0], rgb_max[0])
            g = np.random.randint(rgb_min[1], rgb_max[1])
            b = np.random.randint(rgb_min[2], rgb_max[2])
            data.append({'soil_type': soil_type, 'r': r, 'g': g, 'b': b})
        df = pd.DataFrame(data)
        os.makedirs('../uploads/synthetic_data', exist_ok=True)
        df.to_csv('../uploads/synthetic_data/soil_rgb_data.csv', index=False)
        # Convert to images
        os.makedirs('../uploads/train_images', exist_ok=True)
        for idx, row in df.iterrows():
            img = np.ones((224, 224, 3), dtype=np.uint8)
            img[:, :] = [row['r'], row['g'], row['b']]
            soil_dir = os.path.join('../uploads/train_images', row['soil_type'])
            os.makedirs(soil_dir, exist_ok=True)
            cv2.imwrite(os.path.join(soil_dir, f'image_{idx}.jpg'), img)
        return df

    def generate_synthetic_crop_data(num_samples=1000):
        soil_types = ['Red', 'Alluvial', 'Black', 'Clay', 'Loam', 'Silt', 'Sandy']
        crops = ['Rice', 'Wheat', 'Cotton', 'Maize', 'Soybean', 'Barley', 'Sorghum']
        data = []
        for _ in range(num_samples):
            soil_type = np.random.choice(soil_types)
            pH = np.random.uniform(4.5, 8.5)
            rainfall = np.random.uniform(500, 1500)
            humidity = np.random.uniform(40, 90)
            if soil_type in ['Alluvial', 'Loam'] and pH >= 6.0 and rainfall >= 800:
                crop = 'Rice'
            elif soil_type in ['Red', 'Loam', 'Sandy'] and pH >= 5.5 and rainfall >= 600:
                crop = 'Wheat'
            elif soil_type in ['Black', 'Clay'] and humidity >= 60:
                crop = 'Cotton'
            elif soil_type in ['Sandy', 'Loam'] and rainfall >= 700:
                crop = 'Maize'
            elif soil_type in ['Red', 'Silt'] and pH <= 7.0:
                crop = 'Soybean'
            elif soil_type in ['Alluvial', 'Silt'] and rainfall <= 1000:
                crop = 'Barley'
            else:
                crop = 'Sorghum'
            data.append({
                'soil_type': soil_type,
                'pH': round(pH, 2),
                'rainfall': round(rainfall, 2),
                'humidity': round(humidity, 2),
                'crop': crop
            })
        df = pd.DataFrame(data)
        os.makedirs('../uploads/synthetic_data', exist_ok=True)
        df.to_csv('../uploads/synthetic_data/crop_data.csv', index=False)
        return df

    # --- Routes ---
    @app.route('/')
    def index():
        return redirect(url_for('officer_dashboard'))

    @app.route('/officer/dashboard')
    def officer_dashboard():
        return render_template('admin_dashboard.html')

    @app.route('/officer/monitor_trends')
    def monitor_trends():
        return render_template('monitor_trends.html')

    @app.route('/officer/upload_soil_image', methods=['GET', 'POST'])
    def upload_soil_image():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file uploaded', 'danger')
                return redirect(url_for('monitor_trends'))
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'danger')
                return redirect(url_for('monitor_trends'))
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                image_path = os.path.join('../uploads', filename)
                os.makedirs('../uploads', exist_ok=True)
                file.save(image_path)
                soil_type = predict_soil_type(image_path)
                soil_type_encoded = soil_encoder.transform([soil_type])[0]
                rgb = extract_rgb(image_path)
                pH = 6.5
                input_data = np.array([[soil_type_encoded, pH, 800, 65]])
                crop_pred = rf_model.predict(input_data)
                crop = crop_encoder.inverse_transform(crop_pred)[0]
                flash('Analysis completed successfully', 'success')
                return render_template('view_analyses.html', soil_type=soil_type, pH=pH, crop=crop)
            else:
                flash('Unsupported file type', 'danger')
                return redirect(url_for('monitor_trends'))
        return render_template('upload_soil_image.html')

    @app.route('/officer/previous_analyses')
    def previous_analyses():
        return render_template('previous_analyses.html')

    @app.route('/officer/generate_report')
    def generate_report():
        return render_template('generate_report.html')

    @app.route('/officer/educational_content_add')
    def educational_content_add():
        return render_template('educational_content_add.html')

    @app.route('/officer/manage_users')
    def manage_users():
        return render_template('manage_users.html')

    @app.route('/officer/system_backup')
    def system_backup():
        return render_template('system_backup.html')

    @app.route('/admin/dashboard')
    def admin_dashboard():
        return render_template('admin_dashboard.html')

    @app.route('/admin/train_model', methods=['GET', 'POST'])
    def train_model():
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file uploaded, generating and training with synthetic data', 'info')
                # Generate synthetic data
                soil_df = generate_synthetic_soil_data(num_samples=1000)
                crop_df = generate_synthetic_crop_data(num_samples=1000)
            else:
                file = request.files['file']
                if file.filename == '':
                    flash('No file selected, generating and training with synthetic data', 'info')
                    soil_df = generate_synthetic_soil_data(num_samples=1000)
                    crop_df = generate_synthetic_crop_data(num_samples=1000)
                elif file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join('../uploads', filename)
                    os.makedirs('../uploads', exist_ok=True)
                    file.save(file_path)
                    # Assume file is a CSV or directory; for simplicity, use synthetic data as fallback
                    soil_df = generate_synthetic_soil_data(num_samples=1000)
                    crop_df = generate_synthetic_crop_data(num_samples=1000)
                else:
                    flash('Unsupported file type', 'danger')
                    return redirect(url_for('train_model'))

            # Train Soil Model (CNN)
            datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
            train_generator = datagen.flow_from_directory(
                '../uploads/train_images',
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='training'
            )
            val_generator = datagen.flow_from_directory(
                '../uploads/train_images',
                target_size=(224, 224),
                batch_size=32,
                class_mode='categorical',
                subset='validation'
            )
            soil_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            soil_model.fit(train_generator, validation_data=val_generator, epochs=5)
            soil_model.save('../uploads/soil_model.h5')
            flash('Soil classification model trained and saved as soil_model.h5', 'success')

            # Train Crop Model (Random Forest)
            crop_df['soil_type'] = soil_encoder.fit_transform(crop_df['soil_type'])
            crop_df['crop'] = crop_encoder.fit_transform(crop_df['crop'])
            X = crop_df[['soil_type', 'pH', 'rainfall', 'humidity']]
            y = crop_df['crop']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model.fit(X_train, y_train)
            flash('Crop recommendation model trained', 'success')

        return render_template('train_model.html')

    @app.route('/admin/educational_content', methods=['GET', 'POST'])
    def admin_educational_content():
        contents = []
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file uploaded', 'danger')
                return redirect(url_for('admin_educational_content'))
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'danger')
                return redirect(url_for('admin_educational_content'))
            if file and allowed_file(file.filename):
                print(f"File accepted: {file.filename}, type: {file.content_type}")
                filename = secure_filename(file.filename)
                file_path = os.path.join('../uploads', filename)
                os.makedirs('../uploads', exist_ok=True)
                file.save(file_path)
                contents.append({'id': len(contents) + 1, 'filename': filename, 'content_type': file.content_type})
                flash('File uploaded successfully', 'success')
            else:
                print(f"File rejected: {file.filename}")
                flash('Unsupported file type', 'danger')
        return render_template('educational_content.html', contents=contents)

    @app.route('/admin/download/<content_id>')
    def admin_download_content(content_id):
        contents = []
        content = next((c for c in contents if str(c['id']) == content_id), None)
        if content:
            file_path = os.path.join('../uploads', content['filename'])
            return send_file(file_path, as_attachment=True, mimetype=content['content_type'])
        flash('File not found', 'danger')
        return redirect(url_for('admin_educational_content'))

    @app.route('/admin/content')
    def admin_content():
        return render_template('admin_content.html')