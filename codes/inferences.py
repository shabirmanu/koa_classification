from torchvision import transforms
from main import build_model
import torch
# Constants.
DATA_PATH = 'D:\\Gachon study\\Masters\\Masters\\1st sem\\Osteo\\test\\test'
IMAGE_SIZE = 224
DEVICE = 'cpu'

# Class names.
class_names = ['0', '1', '2', '3', '4']

# Load the trained model.
model = build_model(pretrained=False, fine_tune=False, num_classes=5)
checkpoint = torch.load(
    'D:\\Gachon study\\Masters\\Masters\\1st sem\\Computer Vision\\Osteo\\Koa\\model\\output\\ftloss_pretrained_final1False.pth',
    map_location=DEVICE)
print('Loading trained model weights...')
model.load_state_dict(checkpoint['model_state_dict'])

# Get all the test image paths.
all_image_paths = glob.glob(f"{DATA_PATH}/*")
# Iterate over all the images and do forward pass.
for image_path in all_image_paths:
    # Get the ground truth class name from the image path.
    gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0]
    # Read the image and create a copy.
    image = cv2.imread(image_path)
    orig_image = image.copy()

    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6034, 0.6034, 0.6034],
            std=[0.1596, 0.1596, 0.1596]
        )
    ])
    image = transform(image)
    image = torch.unsqueeze(image, 0)
    image = image.to(DEVICE)

    # Forward pass throught the image.
    outputs = model(image)
    outputs = outputs.detach().numpy()
    pred_class_name = class_names[np.argmax(outputs[0])]
    print(f"GT: {gt_class_name}, Pred: {pred_class_name.lower()}")
    # Annotate the image with ground truth.
    cv2.putText(
        orig_image, f"GT: {gt_class_name}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 255, 0), 2, lineType=cv2.LINE_AA
    )
    # Annotate the image with prediction.
    cv2.putText(
        orig_image, f"Pred: {pred_class_name.lower()}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (100, 100, 225), 2, lineType=cv2.LINE_AA
    )
    cv2.imshow('Result', orig_image)
    cv2.waitKey(0)
    cv2.imwrite(f"D:\\Gachon study\\Masters\\Masters\\1st sem\\Osteo\\test\\test\\tn{gt_class_name}.png", orig_image)