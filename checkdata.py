
import os
import csv
import shutil

from setting import Task, TRAINING_DATA_PATH, SCREEN_HEIGHT, SCREEN_WIDTH
from TapVisualizer import TapVisualizer


# === Loading datas ===
training_data1 = os.path.join(TRAINING_DATA_PATH, "data2025-06-22.csv")
data_sets = [training_data1]

all_rows = []

for data_path in data_sets:
    with open(data_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=[
            'image_path', 'task', 'x_norm', 'y_norm', 'done', 'confidence'])
        for row in reader:
            # Convert or process if needed
            row['x_norm'] = float(row['x_norm'])
            row['y_norm'] = float(row['y_norm'])
            row['done'] = int(row['done'])
            row['confidence'] = int(row['confidence'])
            # Just keep row as dict, or do other processing

            # Append to your list, NOT the file path string
            all_rows.append(row)


# === Save data paths ===
directory = "train_data"
image_path = os.path.join(directory, "images")
data_path = os.path.join(directory, "data")
csv_name = os.path.join(data_path, "clean_data.csv")

invalid_directory = "invalid_data"
invalid_image_path = os.path.join(invalid_directory, "images")
invalid_data_path = os.path.join(invalid_directory, "data")
invalid_csv_name = os.path.join(invalid_data_path, "invalid_data.csv")

os.makedirs(image_path, exist_ok=True)
os.makedirs(data_path, exist_ok=True)
os.makedirs(invalid_image_path, exist_ok=True)
os.makedirs(invalid_data_path, exist_ok=True)

# === Create file if not exists ===
file_exists = os.path.exists(csv_name)

if not file_exists:
    with open(csv_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header only once when file is new
        writer.writerow(["image_path", "task", "x_norm",
                        "y_norm", "done", "confidence"])

file_exists = os.path.exists(invalid_csv_name)

if not file_exists:
    with open(invalid_csv_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write header only once when file is new
        writer.writerow(["image_path", "task", "x_norm",
                        "y_norm", "done", "confidence"])

# === helper function ===


def save_data(row):
    with open(csv_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            row['image_path'],
            row['task'],
            row['x_norm'],
            row['y_norm'],
            row['done'],
            row['confidence']
        ])


def ban_data(row):
    with open(invalid_csv_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow([
            row['image_path'],
            row['task'],
            row['x_norm'],
            row['y_norm'],
            row['done'],
            row['confidence']
        ])


# === main logic ===
valid_count = 0

with open(csv_name, mode="r", newline="", encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # subtract 1 if there is a header
    valid_count = sum(1 for row in reader) - 1

print(f"{valid_count} datas exists in {csv_name}, continue valid data with new image name: image_{(valid_count+1):06d}")

invalid_count = 0

with open(invalid_csv_name, mode="r", newline="", encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    # subtract 1 if there is a header
    invalid_count = sum(1 for row in reader) - 1

print(f"{invalid_count} datas exists in {invalid_csv_name}, continue incalid data with new image name: image_{(invalid_count+1):06d}")

print("Task:")
for task in Task:
    print(f"{task.name:25s} : {task.value}")

while True:
    user_input = input("Please enter Task value: ")

    # Validate input is a digit
    if not user_input.isdigit():
        print("Invalid input, please enter a number.")
        continue

    task_val = int(user_input)

    # Validate input is within enum values
    if task_val not in [t.value for t in Task]:
        print("Invalid input, please try again.")
        continue

    print(f"You selected task value: {task_val}")
    break

checking_row = []

for row in all_rows:
    if int(row['task']) == task_val:
        checking_row.append(row)

print(f"Total data related to {Task(task_val)}: {len(checking_row)}")

visualizer = TapVisualizer()

for row in checking_row:
    visualizer.change_image(row['image_path'])
    task = Task(int(row['task']))
    if row['done'] == 1:
        visualizer.draw_text(text=f"{task.name} Done?", color='green')
        prompt = (f"[{task.name}] Is this Image done?")
    elif row['confidence'] == 1:
        x = row['x_norm'] * SCREEN_WIDTH
        y = row['y_norm']*SCREEN_HEIGHT
        visualizer.add_tap(x, y)
        prompt = (f"[{task.name}] Is this tap correct?")
    elif row['confidence'] == 0:
        visualizer.draw_text(text=f"{task.name} Skip?", color='yellow')
        prompt = (f"[{task.name}] Is this skip correct?")

    prompt += f"\t y/n: "
    user_input = input(prompt).strip().lower()

    if user_input == 'y' or user_input == '':
        print("Valid data, move to new path and write to csv")
        # copy image to new folder
        valid_count += 1
        new_filename = f"image_{(valid_count):06d}.png"

        os.makedirs(image_path, exist_ok=True)

        destination_path = os.path.join(
            image_path, os.path.basename(new_filename))

        shutil.copy2(row['image_path'], destination_path)

        # modifying new data match new dir
        row['image_path'] = os.path.join(image_path, new_filename)

        save_data(row)
        continue

    elif user_input == 'q':
        print("User quitting...")
        break
    elif user_input == 'n':
        pass
    else:
        print("Invalid input treat as invalid data")

    print("Invalid data, move to new path and write to csv")
    # copy image to new folder
    invalid_count += 1
    new_filename = f"image_{(invalid_count):06d}.png"

    os.makedirs(invalid_image_path, exist_ok=True)

    destination_path = os.path.join(
        invalid_image_path, os.path.basename(new_filename))

    shutil.copy2(row['image_path'], destination_path)

    # modifying new data match new dir
    row['image_path'] = os.path.join(invalid_image_path, new_filename)

    ban_data(row)
