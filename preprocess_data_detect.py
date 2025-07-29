import glob
import os
import random
import shutil

def split_dataset(dataset_path, split_ratio=0.8):
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    # Lấy tất cả ảnh jpg
    all_images = list(glob.iglob(os.path.join(images_path, "*.jpg")))
    random.shuffle(all_images)

    train_size = int(len(all_images) * split_ratio)
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]

    # Tạo các thư mục đích nếu chưa có
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)

    def move_images_and_labels(image_list, split_type):
        for image_path in image_list:
            file_name = os.path.basename(image_path)
            label_name = file_name.replace('.jpg', '.txt')

            # Di chuyển ảnh
            target_image_path = os.path.join(dataset_path, f'images/{split_type}', file_name)
            shutil.move(image_path, target_image_path)

            # Di chuyển label (nếu tồn tại)
            label_source = os.path.join(labels_path, label_name)
            label_target_dir = os.path.join(dataset_path, f'labels/{split_type}')
            label_target_path = os.path.join(label_target_dir, label_name)

            if os.path.exists(label_source):
                shutil.move(label_source, label_target_path)
            else:
                print(f"[!] ⚠ Không tìm thấy label cho ảnh: {file_name}")

    # Chia tập train/val
    move_images_and_labels(train_images, 'train')
    move_images_and_labels(val_images, 'val')

    print(f"✅ Đã chia: {len(train_images)} ảnh train, {len(val_images)} ảnh val")

if __name__ == "__main__":
    dataset_path = r"C:\Users\Public\Documents\Project_detect_emotion\DetectEmtion"
    split_dataset(dataset_path)
