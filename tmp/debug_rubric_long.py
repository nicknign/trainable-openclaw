"""Test rubric scoring with LONG answers to reproduce the remote failures."""
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-906ad0dc48354e7aba594ef6d9aa5be6")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash"
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

with open("data/coding_debug/rubrics_v2.json", "r", encoding="utf-8") as f:
    rubrics = json.load(f)

# Simulate a LONG answer (8000+ chars like the real model output)
long_code = """Here is a complete implementation of the requested functionality:

```python
import pygame
import math
import sys
from datetime import datetime

# Initialize Pygame
pygame.init()

# Constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
CENTER = (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
CLOCK_RADIUS = 200
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)

class Clock:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Analog Clock")
        self.clock = pygame.time.Clock()
        self.running = True
        self.font = pygame.font.Font(None, 36)

    def draw_hand(self, angle, length, color, width=2):
        end_x = CENTER[0] + length * math.sin(math.radians(angle))
        end_y = CENTER[1] - length * math.cos(math.radians(angle))
        pygame.draw.line(self.screen, color, CENTER, (int(end_x), int(end_y)), width)

    def draw_clock_face(self):
        pygame.draw.circle(self.screen, WHITE, CENTER, CLOCK_RADIUS, 0)
        pygame.draw.circle(self.screen, BLACK, CENTER, CLOCK_RADIUS, 5)
        pygame.draw.circle(self.screen, BLACK, CENTER, 5, 0)

        for i in range(12):
            angle = math.radians(i * 30 - 90)
            x = CENTER[0] + (CLOCK_RADIUS - 30) * math.cos(angle)
            y = CENTER[1] + (CLOCK_RADIUS - 30) * math.sin(angle)
            number_text = self.font.render(str(i if i != 0 else 12), True, BLACK)
            text_rect = number_text.get_rect(center=(int(x), int(y)))
            self.screen.blit(number_text, text_rect)

        for i in range(60):
            angle = math.radians(i * 6 - 90)
            if i % 5 == 0:
                start_radius = CLOCK_RADIUS - 15
                end_radius = CLOCK_RADIUS
            else:
                start_radius = CLOCK_RADIUS - 5
                end_radius = CLOCK_RADIUS
            start_x = CENTER[0] + start_radius * math.cos(angle)
            start_y = CENTER[1] + start_radius * math.sin(angle)
            end_x = CENTER[0] + end_radius * math.cos(angle)
            end_y = CENTER[1] + end_radius * math.sin(angle)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def draw_time(self, current_time):
        time_str = current_time.strftime("%H:%M:%S")
        text = self.font.render(time_str, True, BLACK)
        text_rect = text.get_rect(center=(CENTER[0], CENTER[1] + CLOCK_RADIUS + 40))
        self.screen.blit(text, text_rect)

    def run(self):
        while self.running:
            self.handle_events()
            self.screen.fill(GRAY)
            self.draw_clock_face()

            now = datetime.now()
            hours = now.hour % 12
            minutes = now.minute
            seconds = now.second

            hour_angle = (hours + minutes / 60) * 30 - 90
            minute_angle = minutes * 6 - 90
            second_angle = seconds * 6 - 90

            self.draw_hand(hour_angle, CLOCK_RADIUS * 0.5, BLACK, 6)
            self.draw_hand(minute_angle, CLOCK_RADIUS * 0.7, BLUE, 4)
            self.draw_hand(second_angle, CLOCK_RADIUS * 0.8, RED, 2)

            self.draw_time(now)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    clock = Clock()
    clock.run()
```

This implementation creates a fully functional analog clock with hour, minute, and second hands, proper numbering, tick marks, and a digital time display below the clock face. The clock updates in real-time using Pygame's event loop and rendering system.""" * 2  # Double to get ~8000 chars

print(f"Long answer length: {len(long_code)} chars")

# Test each rubric
for i, rubric in enumerate(rubrics):
    prompt = rubric['评分提示词'].replace('{content}', long_code)
    print(f"\nRubric {i+1}: {rubric['名称']} | Prompt len={len(prompt)}")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500,
        extra_body={"thinking": {"type": "disabled"}},
    )
    raw = resp.choices[0].message.content
    print(f"  Response length: {len(raw)}")

    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if len(lines) > 1 else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        print(f"  SUCCESS: score={result.get('分数', '?')}, summary={result.get('总结', '')[:80]}")
    except json.JSONDecodeError:
        print(f"  FAILED: raw[:150]={raw[:150]}")
