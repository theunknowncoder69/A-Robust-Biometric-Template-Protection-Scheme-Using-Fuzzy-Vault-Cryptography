# ============================================
# 1. IMPORTS AND GLOBAL VARIABLES
# ============================================

import numpy as np
import cv2
from skimage.morphology import skeletonize
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import time
import matplotlib.pyplot as plt
import warnings
import tkinter as tk
from tkinter import filedialog, messagebox
from sklearn.model_selection import GridSearchCV
from PIL import Image, ImageTk

warnings.filterwarnings("ignore")

BLOCK_SIZE = 16
R = 20           # Number of Minutiae points
S = 180          # Number of Chaff points

# ============================================
# 2. PREPROCESSING AND FEATURE EXTRACTION
# ============================================
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Impostor"], yticklabels=["Genuine", "Impostor"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def do_segmentation(img):
    shape = img.shape
    seg_mask = np.ones(shape)
    threshold = np.var(img) * 0.1
    for i in range(0, img.shape[0], BLOCK_SIZE):
        for j in range(0, img.shape[1], BLOCK_SIZE):
            x, y = min(img.shape[0], i + BLOCK_SIZE), min(img.shape[1], j + BLOCK_SIZE)
            if np.var(img[i:x, j:y]) <= threshold:
                seg_mask[i:x, j:y] = 0
    return np.where(seg_mask == 0, 255, img)

def do_normalization(segmented_image):
    desired_mean, desired_variance = 100.0, 8000.0
    mean, variance = np.mean(segmented_image), np.var(segmented_image)
    normalized_image = (segmented_image - mean) * (desired_variance / variance)**0.5 + desired_mean
    return cv2.normalize(normalized_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def do_thinning(binarized_image):
    return np.where(skeletonize(binarized_image / 255), 0, 1).astype(np.uint8)

def minutiae_points_computer(img):
    segmented = do_segmentation(img)
    normalized = do_normalization(segmented)
    _, binarized = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thinned = do_thinning(binarized)
    minutiae_points = {
        (i, j): (0, 1) for i in range(thinned.shape[0]) for j in range(thinned.shape[1]) if thinned[i, j] == 0
    }
    return minutiae_points

def generate_feature_vector(minutiae_points):
    return np.array([[x, y, orientation] for (x, y), (_, orientation) in minutiae_points.items()]).flatten()

def pad_feature_vectors(feature_vectors, max_length):
    return np.array(
        [np.pad(vector, (0, max_length - len(vector))) if len(vector) < max_length else vector[:max_length] for vector in feature_vectors],
        dtype=np.int32  # Use a smaller data type
    )
# ============================================
# 3. DATASET PREPARATION
# ============================================

def load_dataset(dataset_dir, is_fingerprint=True):
    feature_vectors, labels = [], []
    label = 0

    subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    if not subdirs:
        raise ValueError(f"No subdirectories found in {dataset_dir}. Ensure the directory is structured correctly.")

    for subdir in sorted(subdirs):
        subdir_path = os.path.join(dataset_dir, subdir)
        for img_file in sorted(os.listdir(subdir_path)):
            img_path = os.path.join(subdir_path, img_file)
            if not img_file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                print(f"Skipping non-image file: {img_path}")
                continue

            img = cv2.imread(img_path, 0)
            if img is None:
                print(f"Invalid or unreadable image: {img_path}")
                continue

            try:
                print(f"Processing image: {img_path}")
                minutiae_points = minutiae_points_computer(img)
                if not minutiae_points:
                    print(f"No minutiae points detected: {img_path}")
                    continue
                feature_vector = generate_feature_vector(minutiae_points)
                if feature_vector.size > 0:
                    feature_vectors.append(feature_vector)
                    labels.append(label)
                else:
                    print(f"Empty feature vector: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        label += 1

    if not feature_vectors:
        raise ValueError(f"No valid feature vectors generated from dataset in {dataset_dir}. Check your images or processing pipeline.")

    max_length = max(len(v) for v in feature_vectors)
    feature_vectors = pad_feature_vectors(feature_vectors, max_length)

    return np.array(feature_vectors), np.array(labels)

# ============================================
# 4. TRAINING AND RECOGNITION
# ============================================

def train_recognition_system(X_train, y_train):
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        raise ValueError(f"Training data must contain at least two classes. Found {len(unique_classes)} class(es).")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    param_grid = {
    'n_estimators': [50, 100, 200],  
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
    }



    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    clf_best = grid_search.best_estimator_
    joblib.dump(clf_best, 'rf_recognition_model.pkl')
    return clf_best

from sklearn.metrics import roc_curve

def compute_frr_far_eer(y_true, y_scores):
    # Ensure y_scores is 1D: Select only the column corresponding to the positive class
    if y_scores.ndim > 1:
        y_scores = y_scores[:, 1]  # Take the second column (positive class probability)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr  # False Negative Rate (FRR)

    # Find EER: Point where FPR == FNR (approximately)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_index]
    eer = (fpr[eer_index] + fnr[eer_index]) / 2

    print(f"False Acceptance Rate (FAR) at threshold {eer_threshold:.4f}: {fpr[eer_index]:.4f}")
    print(f"False Rejection Rate (FRR) at threshold {eer_threshold:.4f}: {fnr[eer_index]:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f}")

    return fpr[eer_index], fnr[eer_index], eer

from sklearn.metrics.pairwise import cosine_similarity

def cosine_match(X_train, X_test, y_train):
    similarities = cosine_similarity(X_test, X_train)
    predictions = [y_train[np.argmax(sim)] for sim in similarities]
    return np.array(predictions)

def evaluate_recognition_system(clf, X_test, y_test, dataset_type, X_train, y_train):
    # Classifier-based prediction
    clf_pred = clf.predict(X_test)

    # Cosine similarity-based prediction
    cosine_pred = cosine_match(X_train, X_test, y_train)

    # Combine predictions (majority vote or trust cosine similarity more)
    y_pred = np.where(clf_pred == cosine_pred, clf_pred, cosine_pred)

    accuracy = accuracy_score(y_test, y_pred) * 100

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix for {dataset_type}:")
    print(cm)

    # Classification Report
    print(f"\nClassification Report for {dataset_type}:")
    print(classification_report(y_test, y_pred))

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {dataset_type}")
    plt.show()

    return accuracy

# ============================================
# 5. GUI, PLOT AND MAIN FUNCTION
# ============================================

def show_accuracy_window(accuracy, dataset_type):
    window = tk.Tk()
    window.title("Recognition Accuracy")
    window.geometry("1000x600")

    background_image_path = "Images/Template Protection.png"
    try:
        bg_image = Image.open(background_image_path)
    except FileNotFoundError:
        print(f"Error: Background image '{background_image_path}' not found.")
        return

    canvas = tk.Canvas(window, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    text_id = None
    close_button_window = None

    def resize_elements(event):
        nonlocal text_id, close_button_window

        new_width, new_height = event.width, event.height
        resized_image = bg_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        canvas.bg_photo = ImageTk.PhotoImage(resized_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=canvas.bg_photo)
        text_id = canvas.create_text(
            new_width // 2, int(new_height * 0.8),
            text=f"{dataset_type} Recognition Accuracy\n{accuracy:.2f}%",
            font=("Arial", 22),
            fill="white",
            anchor="center"
        )
        if close_button_window is None:
            close_button = tk.Button(
                window,
                text="Close",
                command=window.destroy,
                font=("Arial", 12),
                bg="#61dafb",
                fg="#282c34",
                padx=10,
                pady=5
            )
            close_button_window = canvas.create_window(new_width // 2, new_height - 50, anchor="center", window=close_button)
        else:
            canvas.coords(close_button_window, new_width // 2, new_height - 50)
    window.bind("<Configure>", resize_elements)

    window.mainloop()
    
def plot_minutiae_points(img, minutiae_points):
    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap='gray')
    minutiae_x = [y for x, y in minutiae_points]
    minutiae_y = [x for x, y in minutiae_points]
    plt.scatter(minutiae_x, minutiae_y, color='red', s=10)
    plt.title("Minutiae Points on Fingerprint")
    plt.show()

def select_fingerprint_and_plot():
    file_path = filedialog.askopenfilename(title="Select Fingerprint Image", filetypes=[("Image Files", ".bmp;.png;.jpg;.jpeg;*.tif")])
    if file_path:
        img = cv2.imread(file_path, 0)
        minutiae_points = minutiae_points_computer(img)
        plot_minutiae_points(img, minutiae_points)
    else:
        messagebox.showwarning("File Selection", "No file selected. Please select a valid fingerprint image.")

def main():
    start_time = time.time()
    base_fp_dir = os.path.join("fingerprint_dataset", "SOCOFing")
    train_fp_dirs = ["Train"]
    test_fp_dir = os.path.join(base_fp_dir, "Test")
    print("Loading fingerprint training datasets...")
    fp_train_features, fp_train_labels = [], []
    all_train_features = []

    for train_dir in train_fp_dirs:
        full_train_dir = os.path.join(base_fp_dir, train_dir)
        load_start = time.time()  # Measure dataset loading time
        features, labels = load_dataset(full_train_dir, is_fingerprint=True)
        load_end = time.time()
        print(f"Dataset loading time: {load_end - load_start:.2f} seconds")
        max_length_fp = max([len(vector) for vector in features])
        padded_features = pad_feature_vectors(features, max_length_fp)
        
        fp_train_features.append(padded_features)
        fp_train_labels.append(labels)
        all_train_features.extend(padded_features)  # Flatten to get all features together

    max_length_fp = max([len(vector) for vector in all_train_features])
    X_train_fp = np.concatenate([pad_feature_vectors(features, max_length_fp) for features in fp_train_features], axis=0)
    y_train_fp = np.concatenate(fp_train_labels, axis=0)
    print("Loading fingerprint testing dataset...")
    test_load_start = time.time()
    X_test_fp, y_test_fp = load_dataset(test_fp_dir, is_fingerprint=True)
    test_load_end = time.time()
    print(f"Testing dataset loading time: {test_load_end - test_load_start:.2f} seconds")
    X_test_fp = pad_feature_vectors(X_test_fp, max_length_fp)
    print("Normalizing features...")
    norm_start = time.time()
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    X_train_fp = scaler.fit_transform(X_train_fp)
    X_test_fp = scaler.transform(X_test_fp)
    
 
    norm_end = time.time()
    print(f"Feature normalization time: {norm_end - norm_start:.2f} seconds") 
    print("Training fingerprint recognition system...")
    train_start = time.time()
    clf_fp = train_recognition_system(X_train_fp, y_train_fp)
    train_end = time.time()
    print(f"Training time: {train_end - train_start:.2f} seconds")
    print("Evaluating fingerprint recognition system...")
    eval_start = time.time()
    accuracy = evaluate_recognition_system(clf_fp, X_test_fp, y_test_fp, "Fingerprint", X_train_fp, y_train_fp)
    print(f"Fingerprint Recognition Accuracy: {accuracy:.2f}%")
    eval_end = time.time()
    print(f"Evaluation time: {eval_end - eval_start:.2f} seconds")
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")
root = tk.Tk()
root.title("Fingerprint Recognition System")
plot_button = tk.Button(root, text="Select and Plot Minutiae Points", command=select_fingerprint_and_plot)
plot_button.pack(padx=20, pady=20)
root.mainloop()

if __name__ == "__main__":
    main()