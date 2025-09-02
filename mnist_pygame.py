import pygame
import numpy as np
import os
from datetime import datetime
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.datasets import mnist
from keras.utils import to_categorical

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 600
DRAW_AREA_SIZE = 300
DRAW_AREA_POS = (50, 150)
COLORS = {
    'background': (25, 25, 35),
    'panel': (45, 45, 60),
    'draw_area': (35, 35, 50),
    'button': (80, 180, 220),
    'button_hover': (100, 200, 240),
    'button_text': (240, 245, 255),
    'text': (220, 225, 240),
    'text_dark': (160, 165, 180),
    'correct': (100, 220, 100),
    'incorrect': (220, 100, 100),
    'prediction': (100, 180, 240),
    'confidence_high': (100, 220, 100),
    'confidence_med': (220, 200, 100),
    'confidence_low': (220, 100, 100),
    'accent': (80, 180, 220),
    'dark_accent': (35, 35, 50),
}

# Fonts
try:
    title_font = pygame.font.Font("fonts/pixel.ttf", 48)
    header_font = pygame.font.Font("fonts/pixel.ttf", 32)
    main_font = pygame.font.Font("fonts/pixel.ttf", 24)
    small_font = pygame.font.Font("fonts/pixel.ttf", 18)
except:
    # Fallback to system fonts
    title_font = pygame.font.SysFont('Arial', 48, bold=True)
    header_font = pygame.font.SysFont('Arial', 32, bold=True)
    main_font = pygame.font.SysFont('Arial', 24)
    small_font = pygame.font.SysFont('Arial', 18)

# Setup
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pixel Digit Detective")
clock = pygame.time.Clock()

# Game variables
score = 0
high_score = 0
streak = 0
feedback_data = []
drawing = np.zeros((28, 28), dtype=np.float32)
last_prediction = None
last_confidence = 0
last_correct = None
brush_size = 2  # Increased brush size for better drawing
needs_redraw = True
particles = []

# Directories
MODEL_DIR = "models"
FEEDBACK_DIR = "feedback_data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FEEDBACK_DIR, exist_ok=True)

# Sound effects - with better error handling
try:
    correct_sound = pygame.mixer.Sound("sounds/correct.wav")
    incorrect_sound = pygame.mixer.Sound("sounds/incorrect.wav")
    draw_sound = pygame.mixer.Sound("sounds/draw.wav")
except:
    # Create silent dummy sounds
    correct_sound = pygame.mixer.Sound(buffer=bytearray())
    incorrect_sound = pygame.mixer.Sound(buffer=bytearray())
    draw_sound = pygame.mixer.Sound(buffer=bytearray())

class Button:
    def __init__(self, x, y, width, height, text, action=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.action = action
        self.hovered = False
        
    def draw(self, surface):
        color = COLORS['button_hover'] if self.hovered else COLORS['button']
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        
        text_surf = main_font.render(self.text, True, COLORS['button_text'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)
        
    def check_hover(self, pos):
        self.hovered = self.rect.collidepoint(pos)
        return self.hovered
        
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.hovered:
            if self.action:
                self.action()
            return True
        return False

def draw_loading_screen(progress, message):
    """Draw a loading screen with progress bar"""
    screen.fill(COLORS['background'])
    
    # Draw title
    title_text = title_font.render("Pixel Digit Detective", True, COLORS['accent'])
    screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 150))
    
    # Draw message
    loading_text = main_font.render(message, True, COLORS['text'])
    screen.blit(loading_text, (SCREEN_WIDTH // 2 - loading_text.get_width() // 2, 250))
    
    # Draw progress bar background
    bar_width = 400
    bar_height = 20
    bar_x = (SCREEN_WIDTH - bar_width) // 2
    bar_y = 300
    pygame.draw.rect(screen, COLORS['dark_accent'], (bar_x, bar_y, bar_width, bar_height), border_radius=10)
    
    # Draw progress
    progress_width = int(bar_width * progress)
    if progress_width > 0:
        pygame.draw.rect(screen, COLORS['accent'], (bar_x, bar_y, progress_width, bar_height), border_radius=10)
    
    # Draw percentage
    percent_text = main_font.render(f"{int(progress * 100)}%", True, COLORS['text'])
    screen.blit(percent_text, (bar_x + bar_width // 2 - percent_text.get_width() // 2, 
                              bar_y + bar_height // 2 - percent_text.get_height() // 2))
    
    pygame.display.flip()

def load_or_train_model():
    model_path = os.path.join(MODEL_DIR, 'mnist_cnn.h5')
    try:
        draw_loading_screen(0.1, "Loading model...")
        model = load_model(model_path)
        draw_loading_screen(1.0, "Model loaded!")
        pygame.time.delay(500)
        print("Loaded existing model")
        return model
    except:
        print("Training new model...")
        draw_loading_screen(0.2, "Loading MNIST data...")
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        draw_loading_screen(0.4, "Preprocessing data...")
        # Preprocess data
        X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
        X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        draw_loading_screen(0.6, "Building model...")
        # Lightweight model for real-time performance
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(32, activation='relu'),
            Dense(10, activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                     loss='categorical_crossentropy', 
                     metrics=['accuracy'])
        
        draw_loading_screen(0.7, "Training model...")
        # Train with minimal epochs for demo
        model.fit(X_train, y_train, 
                 epochs=1,
                 batch_size=128,
                 validation_data=(X_test, y_test),
                 verbose=0)
        
        draw_loading_screen(0.9, "Saving model...")
        model.save(model_path)
        draw_loading_screen(1.0, "Model trained!")
        pygame.time.delay(500)
        print("Model trained and saved")
        
        return model

def save_feedback_image(img_array, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{label}_{timestamp}.npy"
    filepath = os.path.join(FEEDBACK_DIR, filename)
    np.save(filepath, img_array)
    feedback_data.append((img_array.copy(), label))

def retrain_on_feedback(model):
    if not feedback_data:
        print("No feedback data to train on")
        return model
    
    # Prepare data
    X = np.array([data[0] for data in feedback_data]).reshape(-1, 28, 28, 1)
    y = to_categorical(np.array([data[1] for data in feedback_data]), 10)
    
    # Train with minimal settings for responsiveness
    model.fit(X, y, 
             epochs=1,
             batch_size=32,
             verbose=0)
    
    # Save updated model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.save(os.path.join(MODEL_DIR, f'mnist_cnn_{timestamp}.h5'))
    
    print(f"Model updated with {len(feedback_data)} feedback samples")
    return model

def draw_confidence_bar(surface, confidence, x, y, width, height):
    """Draw a confidence bar"""
    fill_width = int(width * confidence)
    
    # Determine color based on confidence
    if confidence > 0.7:
        color = COLORS['confidence_high']
    elif confidence > 0.4:
        color = COLORS['confidence_med']
    else:
        color = COLORS['confidence_low']
    
    # Draw background
    pygame.draw.rect(surface, COLORS['dark_accent'], (x, y, width, height), border_radius=3)
    # Draw fill
    if fill_width > 0:
        pygame.draw.rect(surface, color, (x, y, fill_width, height), border_radius=3)
    
    # Draw confidence text
    conf_text = small_font.render(f"{confidence*100:.1f}%", True, COLORS['text'])
    surface.blit(conf_text, (x + width + 10, y + height//2 - 7))

def clear_drawing():
    global drawing, needs_redraw
    drawing.fill(0)
    needs_redraw = True

def predict_digit(model):
    # Only predict if there's something drawn
    if np.max(drawing) < 0.01:
        return None, 0, None
        
    digit = drawing.reshape(1, 28, 28, 1)
    prediction = model.predict(digit, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_digit, confidence, prediction

def handle_correct():
    global score, streak, last_correct, needs_redraw, high_score
    score += 10 + streak * 5
    high_score = max(high_score, score)
    streak += 1
    last_correct = True
    needs_redraw = True
    
    try:
        correct_sound.play()
    except:
        pass

def handle_incorrect(correct_digit):
    global score, streak, last_correct, needs_redraw
    score = max(0, score - 5)
    streak = 0
    last_correct = False
    needs_redraw = True
    save_feedback_image(drawing.copy(), correct_digit)
    
    try:
        incorrect_sound.play()
    except:
        pass

# Create buttons
clear_button = Button(SCREEN_WIDTH - 200, 400, 160, 40, "Clear Canvas", clear_drawing)

# Load model
model = load_or_train_model()

# Main game loop
running = True
while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_clicked = False
    drawing_occurred = False
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            # Train on feedback before exiting
            model = retrain_on_feedback(model)
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_clicked = True
            if clear_button.handle_event(event):
                needs_redraw = True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                clear_drawing()
            elif pygame.K_0 <= event.key <= pygame.K_9:
                # User providing correct label
                correct_digit = event.key - pygame.K_0
                if last_prediction is not None and correct_digit == last_prediction:
                    handle_correct()
                else:
                    handle_incorrect(correct_digit)
                # Clear after answer
                pygame.time.delay(300)
                clear_drawing()
    
    # Drawing
    if pygame.mouse.get_pressed()[0]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        draw_area_rect = pygame.Rect(DRAW_AREA_POS[0], DRAW_AREA_POS[1], DRAW_AREA_SIZE, DRAW_AREA_SIZE)
        
        if draw_area_rect.collidepoint(mouse_x, mouse_y):
            x, y = (mouse_x - DRAW_AREA_POS[0]) // (DRAW_AREA_SIZE // 28), (mouse_y - DRAW_AREA_POS[1]) // (DRAW_AREA_SIZE // 28)
            
            # Draw with the brush (more visible)
            for i in range(-brush_size, brush_size + 1):
                for j in range(-brush_size, brush_size + 1):
                    if 0 <= x+i < 28 and 0 <= y+j < 28:
                        dist = (i**2 + j**2)**0.5
                        if dist <= brush_size:
                            drawing[y+j, x+i] = min(1.0, drawing[y+j, x+i] + 0.5)  # Increased drawing strength
            
            drawing_occurred = True
            needs_redraw = True
            
            # Occasionally play draw sound
            if np.random.random() < 0.1:
                try:
                    draw_sound.play()
                except:
                    pass
    
    # Prediction (only if something is drawn or changed)
    if drawing_occurred or (needs_redraw and np.any(drawing > 0)):
        last_prediction, last_confidence, all_predictions = predict_digit(model)
    
    # Only redraw when necessary
    if needs_redraw:
        # Draw background
        screen.fill(COLORS['background'])
        
        # Draw title
        title_text = title_font.render("Pixel Digit Detective", True, COLORS['accent'])
        screen.blit(title_text, (SCREEN_WIDTH // 2 - title_text.get_width() // 2, 20))
        
        # Draw drawing area
        pygame.draw.rect(screen, COLORS['draw_area'], 
                        (DRAW_AREA_POS[0], DRAW_AREA_POS[1], DRAW_AREA_SIZE, DRAW_AREA_SIZE), border_radius=8)
        pygame.draw.rect(screen, COLORS['accent'], 
                        (DRAW_AREA_POS[0], DRAW_AREA_POS[1], DRAW_AREA_SIZE, DRAW_AREA_SIZE), 2, border_radius=8)
        
        # Show drawing
        pixel_size = DRAW_AREA_SIZE // 28
        for y_pos in range(28):
            for x_pos in range(28):
                if drawing[y_pos, x_pos] > 0:
                    alpha = min(255, int(drawing[y_pos, x_pos] * 255))
                    color = (255, 255, 255, alpha)
                    s = pygame.Surface((pixel_size, pixel_size), pygame.SRCALPHA)
                    s.fill(color)
                    screen.blit(s, (DRAW_AREA_POS[0] + x_pos * pixel_size, DRAW_AREA_POS[1] + y_pos * pixel_size))
        
        # Draw right panel
        panel_rect = pygame.Rect(SCREEN_WIDTH - 300, 80, 280, 500)
        pygame.draw.rect(screen, COLORS['panel'], panel_rect, border_radius=10)
        pygame.draw.rect(screen, COLORS['accent'], panel_rect, 2, border_radius=10)
        
        # Draw panel header
        header_text = header_font.render("Game Stats", True, COLORS['text'])
        screen.blit(header_text, (SCREEN_WIDTH - 300 + 140 - header_text.get_width()//2, 100))
        
        # Draw separator
        pygame.draw.line(screen, COLORS['accent'], (SCREEN_WIDTH - 300 + 20, 140), (SCREEN_WIDTH - 20, 140), 2)
        
        # Score display
        score_text = main_font.render(f"Score: {score}", True, COLORS['text'])
        screen.blit(score_text, (SCREEN_WIDTH - 280, 160))
        
        # High score
        high_score_text = small_font.render(f"High Score: {high_score}", True, COLORS['text_dark'])
        screen.blit(high_score_text, (SCREEN_WIDTH - 280, 190))
        
        # Streak display
        streak_text = main_font.render(f"Streak: {streak}", True, COLORS['text'])
        screen.blit(streak_text, (SCREEN_WIDTH - 280, 220))
        
        # Prediction display
        if last_prediction is not None:
            # Show prediction
            pred_text = header_font.render(f"Prediction: {last_prediction}", True, COLORS['prediction'])
            screen.blit(pred_text, (SCREEN_WIDTH - 280, 260))
            
            # Confidence bar
            conf_label = small_font.render("Confidence:", True, COLORS['text'])
            screen.blit(conf_label, (SCREEN_WIDTH - 280, 300))
            draw_confidence_bar(screen, last_confidence, SCREEN_WIDTH - 280, 320, 200, 20)
        
        # Feedback display
        if last_correct is not None:
            if last_correct:
                feedback_text = header_font.render("Correct!", True, COLORS['correct'])
            else:
                feedback_text = header_font.render("Try Again!", True, COLORS['incorrect'])
            screen.blit(feedback_text, (SCREEN_WIDTH - 280, 360))
        
        # Draw clear button
        clear_button.check_hover(mouse_pos)
        clear_button.draw(screen)
        
        # Draw instructions
        instr_text = small_font.render("Draw a digit (0-9) in the canvas", True, COLORS['text_dark'])
        screen.blit(instr_text, (50, 100))
        
        instr_text2 = small_font.render("Press the corresponding number key to score", True, COLORS['text_dark'])
        screen.blit(instr_text2, (50, 470))
        
        instr_text3 = small_font.render("Press C to clear the canvas", True, COLORS['text_dark'])
        screen.blit(instr_text3, (50, 500))
        
        pygame.display.flip()
        needs_redraw = False
    
    clock.tick(60)

pygame.quit()