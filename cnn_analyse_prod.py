import cv2
import numpy as np
from PIL import Image
import os
from typing import List, Optional, Tuple, Dict
import torch
import torch.nn as nn
from torchvision import transforms
import string

# CNN Model Definition
class RobustCNN(nn.Module):
    def __init__(self, num_classes=26):
        super(RobustCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class WordleAnalyzer:
    def __init__(self, model_path: str = 'wordle_cnn_model.pth'):
        """Initialize the Wordle analyzer with trained model"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.transform = None
        self.idx_to_letter = {idx: letter for idx, letter in enumerate(string.ascii_uppercase)}
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load the trained CNN model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = RobustCNN(num_classes=26)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _crop_tile_borders(self, tile: np.ndarray, border_ratio: float = 0.15) -> np.ndarray:
        """Crop the borders of a tile to focus on the letter content"""
        if tile is None or tile.size == 0:
            return tile
        
        h, w = tile.shape[:2]
        crop_h = min(int(h * border_ratio), h // 4)
        crop_w = min(int(w * border_ratio), w // 4)
        
        cropped = tile[crop_h:h-crop_h, crop_w:w-crop_w]
        return cropped if cropped.size > 0 else tile
    
    def _predict_letter(self, tile: np.ndarray) -> str:
        """Predict the letter in a tile using the trained CNN model"""
        if tile is None or tile.size == 0:
            return '?'
        
        try:
            cropped_tile = self._crop_tile_borders(tile)
            tile_rgb = cv2.cvtColor(cropped_tile, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(tile_rgb)
            tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor)
                _, predicted_idx = torch.max(outputs, 1)
                return self.idx_to_letter[predicted_idx.item()]
        except:
            return '?'
    
    def _is_tile_filled(self, tile: np.ndarray) -> bool:
        """Check if a tile is filled (not empty)"""
        if tile is None or tile.size == 0:
            return False
        
        h, w = tile.shape[:2]
        dy, dx = int(0.2*h), int(0.2*w)
        core = tile[dy:h-dy, dx:w-dx]
        if core.size == 0:
            core = tile
        
        hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]
        black_ratio = np.count_nonzero(V < 20) / float(V.size)
        return black_ratio <= 0.95
    
    def _detect_tile_color(self, tile: np.ndarray) -> str:
        """Detect the color of a Wordle tile"""
        if tile is None or tile.size == 0 or not self._is_tile_filled(tile):
            return 'empty'
        
        hsv = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
        h, w = tile.shape[:2]
        center_region = hsv[h//4:3*h//4, w//4:3*w//4]
        
        brightness = center_region[:, :, 2]
        background_mask = (brightness > 20) & (brightness < 220)
        
        if np.count_nonzero(background_mask) < 10:
            return 'gray'
        
        background_pixels = center_region[background_mask]
        avg_hue = np.mean(background_pixels[:, 0])
        avg_saturation = np.mean(background_pixels[:, 1])
        
        if avg_saturation < 80:
            return 'b'
        elif 45 <= avg_hue <= 85:
            return 'g'
        elif 15 <= avg_hue <= 50:
            return 'y' if avg_hue < 35 else 'g'
        else:
            return 'g' if avg_saturation > 100 else 'b'
    
    def _extract_grid(self, image: np.ndarray) -> np.ndarray:
        """Extract the Wordle grid from the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, blockSize=15, C=-2)
        
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horiz = cv2.morphologyEx(th, cv2.MORPH_OPEN, hk)
        vert = cv2.morphologyEx(th, cv2.MORPH_OPEN, vk)
        mask = cv2.bitwise_or(horiz, vert)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            raise RuntimeError("Could not find Wordle grid")
        
        gx, gy, gw, gh = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return image[gy:gy+gh, gx:gx+gw]
    
    def _process_tiles(self, grid_roi: np.ndarray) -> Tuple[List[List[str]], List[List[str]]]:
        """Process all tiles in the grid and return letters and colors"""
        ROWS, COLS = 6, 5
        H, W = grid_roi.shape[:2]
        tile_h, tile_w = H / ROWS, W / COLS
        
        letters = [['.']*COLS for _ in range(ROWS)]
        colors = [['empty']*COLS for _ in range(ROWS)]
        
        for r in range(ROWS):
            for c in range(COLS):
                x1, y1 = int(c * tile_w), int(r * tile_h)
                x2, y2 = int((c + 1) * tile_w), int((r + 1) * tile_h)
                tile = grid_roi[y1:y2, x1:x2]
                
                if self._is_tile_filled(tile):
                    letters[r][c] = self._predict_letter(tile)
                    colors[r][c] = self._detect_tile_color(tile)
        
        return letters, colors
    
    def analyze(self, image_path: str) -> List[Dict[str, str]]:
        """
        Analyze a Wordle screenshot and return completed words with their colors.
        
        Args:
            image_path: Path to the Wordle screenshot
            
        Returns:
            List of dictionaries, each containing:
            - 'word': The completed word
            - 'colors': Space-separated color pattern (e.g., "gray yellow green green gray")
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        grid_roi = self._extract_grid(image)
        letters, colors = self._process_tiles(grid_roi)
        
        results = []
        for r in range(len(letters)):
            # Check if row is complete (all tiles filled)
            if all(letter != '.' for letter in letters[r]):
                word = ''.join(letters[r])
                color_pattern = ''.join(colors[r])
                results.append({
                    'word': word,
                    'colors': color_pattern
                })
        
        return results


# Example usage
def analyze_wordle_screenshot(image_path: str, model_path: str = 'wordle_cnn_model.pth') -> List[Dict[str, str]]:
    """
    Convenience function to analyze a Wordle screenshot.
    
    Args:
        image_path: Path to the Wordle screenshot
        model_path: Path to the trained CNN model
        
    Returns:
        List of dictionaries with completed words and their color patterns
    """
    analyzer = WordleAnalyzer(model_path)
    return analyzer.analyze(image_path)


if __name__ == "__main__":
    # Example usage
    try:
        results = analyze_wordle_screenshot('images/screenshot8.jpg')
        print(results)
        if results:
            print("Completed words:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['word']} -> {result['colors']}")
        else:
            print("No completed words found.")
            
    except Exception as e:
        print(f"Error: {e}")