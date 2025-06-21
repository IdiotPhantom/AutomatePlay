from torchvision.transforms import functional as F
from PIL import Image, ImageOps



class ResizeImage:
    def __init__(self, size=224):
        self.size = size

    def __call__(self, img):
        orig_w, orig_h = img.size
        scale = self.size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)


        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

        img = ImageOps.expand(img, padding, fill=0)  # pad with black
        return F.to_tensor(img)

    def preprocess_image(image, target_size=224):
        orig_w, orig_h = image.size
        scale = target_size / max(orig_w, orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        image = image.resize((new_w, new_h))

        pad_w = target_size - new_w
        pad_h = target_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w -
                   pad_w // 2, pad_h - pad_h // 2)
        image = F.pad(image, padding, fill=0)

        tensor = F.to_tensor(image)
        return tensor.unsqueeze(0), scale, padding

    def unpad_and_rescale_coords(x_norm, y_norm, scale, padding, orig_w, orig_h):
        x_pad = x_norm * 224 - padding[0]
        y_pad = y_norm * 224 - padding[1]

        # Clamp to inside image (between 0 and scaled size)
        x_pad = max(0, min(x_pad, 224 - padding[0] - padding[2]))
        y_pad = max(0, min(y_pad, 224 - padding[1] - padding[3]))

        # Rescale to original coordinates
        x_orig = x_pad / scale
        y_orig = y_pad / scale

        # Clamp to original bounds
        x_orig = max(0, min(x_orig, orig_w))
        y_orig = max(0, min(y_orig, orig_h))
        return int(x_orig), int(y_orig)
