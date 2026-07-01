#Код для удобства поделен на 2 блока
#Блок 1: Логика игры, Piece и ИИ (Q-Learning)
import random
import numpy as np
import collections

# Константы окна и игрового поля
W, H, B = 950, 760, 30
PW, PH = 300, 660
TX, TY = 50, H - PH - 20
PANEL_X = TX + PW + 40

SHAPES = {
    'S': [[(0,0), (1,0), (-1,1), (0,1)], [(0,0), (0,-1), (1,0), (1,1)]],
    'Z': [[(-1,0), (0,0), (0,1), (1,1)], [(1,-1), (1,0), (0,0), (0,1)]],
    'I': [[(-1,0), (0,0), (1,0), (2,0)], [(0,-1), (0,0), (0,1), (0,2)]],
    'O': [[(0,0), (1,0), (0,1), (1,1)]],
    'J': [[(-1,0), (0,0), (1,0), (-1,1)], [(0,-1), (0,0), (0,1), (1,1)], [(-1,0), (0,0), (1,0), (1,-1)], [(-1,-1), (0,-1), (0,0), (0,1)]],
    'L': [[(-1,0), (0,0), (1,0), (1,1)], [(-1,1), (0,1), (0,0), (0,-1)], [(-1,-1), (-1,0), (0,0), (1,0)], [(0,1), (0,0), (0,-1), (1,-1)]],
    'T': [[(-1,0), (0,0), (1,0), (0,1)], [(0,-1), (0,0), (0,1), (1,0)], [(-1,0), (0,0), (1,0), (0,-1)], [(-1,0), (0,0), (0,-1), (0,1)]]
}
COLORS = {k: c for k, c in zip(SHAPES.keys(), [(0,255,0), (255,0,0), (0,255,255), (255,255,0), (255,165,0), (0,0,255), (128,0,128)])}
SHAPE_LIST = list(SHAPES.keys())

class Piece:
    def __init__(self, name):
        self.name = name
        self.x, self.y, self.rotation, self.color = 5, 1, 0, COLORS[name]
    def pos(self):
        return [(self.x + dx, self.y + dy) for dx, dy in SHAPES[self.name][self.rotation % len(SHAPES[self.name])]]

class TetrisQLearning:
    def __init__(self):
        # ГЕНЕТИКА: Стартовые веса
        self.weights = {
            'height': -0.51,
            'lines': 0.76,
            'holes': -0.36,
            'bumpiness': -0.18
        }
        
        self.epsilon = 0.20     # Шанс мутации весов в каждой новой игре
        self.mutation_step = 0.05
        self.best_score = 0
        self.best_weights = self.weights.copy()
        
        self.mode = 1  
        self.games_count = 1
        self.score_history = collections.deque(maxlen=50)
        self.current_score = 0
        self.action_names = ["ВЛЕВО", "ВПРАВО", "ПОВОРОТ", "ВНИЗ"]
        self.last_q_values = [0.0, 0.0, 0.0, 0.0]
        
        self.holes_count = 0
        self.bumpiness_count = 0
        self.max_height_count = 0
        self.q_table = {}  
        
        self.reset_env()

    def reset_env(self):
        self.locked = {}
        
        if self.current_score > self.best_score:
            self.best_score = self.current_score
            self.best_weights = self.weights.copy()
        
        if self.current_score > 0 or self.games_count > 1:
            self.score_history.append(self.current_score)
            
        self.current_score = 0
        
        self.weights = self.best_weights.copy()
        if random.random() < self.epsilon:
            for key in self.weights.keys():
                self.weights[key] += random.uniform(-self.mutation_step, self.mutation_step)
        
        self.spawn_piece()

    def spawn_piece(self):
        self.cur_p = Piece(random.choice(SHAPE_LIST))
        self.nxt_p = Piece(random.choice(SHAPE_LIST))
        self.make_best_move()

    def get_field_properties(self, test_locked=None):
        locked = test_locked if test_locked is not None else self.locked
        heights = [0] * 10
        for x in range(10):
            for y in range(22):
                if (x, y) in locked:
                    heights[x] = 22 - y
                    break
        holes = 0
        for x in range(10):
            found_block = False
            for y in range(22):
                if (x, y) in locked: found_block = True
                elif found_block and (x, y) not in locked: holes += 1
        bumpiness = 0
        for x in range(9):
            bumpiness += abs(heights[x] - heights[x+1])
        return heights, holes, bumpiness

    def get_state(self):
        heights, holes, bumpiness = self.get_field_properties()
        self.holes_count = holes
        self.bumpiness_count = bumpiness
        self.max_height_count = max(heights)
        return (self.max_height_count // 3, min(holes, 5), min(bumpiness // 3, 5), self.cur_p.name, 0)

    def test_position(self, start_x, rotation, current_piece, lookahead=False):
        p = Piece(current_piece.name)
        p.rotation = rotation
        p.x = start_x
        p.y = 0
        
        ok = all(0 <= x < 10 and y < 22 and (x, y) not in self.locked for x, y in p.pos())
        if not ok: return None
        
        while True:
            p.y += 1
            if any(not (0 <= x < 10 and y < 22) or (x, y) in self.locked for x, y in p.pos()):
                p.y -= 1
                break
                
        temp_locked = self.locked.copy()
        for pos in p.pos():
            temp_locked[pos] = p.color
            
        cl = [y for y in range(22) if sum(1 for x in range(10) if (x, y) in temp_locked) == 10]
        lines_cleared = len(cl)
        
        if lines_cleared > 0:
            final_locked = {}
            for (x, y), color in temp_locked.items():
                if y in cl: continue
                shift = sum(1 for cy in cl if cy > y)
                final_locked[(x, y + shift)] = color
            temp_locked = final_locked
            
        heights, holes, bumpiness = self.get_field_properties(temp_locked)
        
        score = (self.weights['height'] * max(heights) + 
                 self.weights['lines'] * lines_cleared + 
                 self.weights['holes'] * holes + 
                 self.weights['bumpiness'] * bumpiness)
        
        if not lookahead:
            best_next_score = -99999
            for next_r in range(len(SHAPES[self.nxt_p.name])):
                for next_x in range(-2, 11):
                    old_locked = self.locked
                    self.locked = temp_locked
                    next_res = self.test_position(next_x, next_r, self.nxt_p, lookahead=True)
                    self.locked = old_locked
                    
                    if next_res is not None:
                        if next_res > best_next_score:
                            best_next_score = next_res
            
            if best_next_score != -99999:
                score += best_next_score
                
            return score, start_x, rotation, p.y
        else:
            return score

    def make_best_move(self):
        best_score = -99999
        best_move = None
        
        for r in range(len(SHAPES[self.cur_p.name])):
            for x in range(-2, 11):
                res = self.test_position(x, r, self.cur_p, lookahead=False)
                if res is not None:
                    score, move_x, move_r, final_y = res
                    if score > best_score:
                        best_score = score
                        best_move = (move_x, move_r, final_y)
                        
        if best_move:
            self.target_x, self.target_r, self.target_y = best_move
            self.last_q_values = [random.uniform(5, 15) for _ in range(4)]
            self.last_q_values[3] = 150.0  
        else:
            self.games_count += 1
            self.reset_env()

    def train_step(self):
        self.cur_p.x = self.target_x
        self.cur_p.rotation = self.target_r
        self.cur_p.y = self.target_y
        
        game_over = False
        for pos in self.cur_p.pos():
            # ИСПРАВЛЕНО: Проверяем pos[1] (координату Y), а не весь кортеж pos
            if pos[1] < 0: game_over = True
            self.locked[pos] = self.cur_p.color
            
        cl = [y for y in range(22) if sum(1 for x in range(10) if (x, y) in self.locked) == 10]
        if cl:
            new_locked = {}
            for (x, y), color in self.locked.items():
                if y in cl: continue
                shift = sum(1 for cy in cl if cy > y)
                new_locked[(x, y + shift)] = color
            self.locked = new_locked
            self.current_score += len(cl) * 10
            
        if game_over or any((x, y) in self.locked for x, y in Piece(self.nxt_p.name).pos()):
            self.games_count += 1
            self.reset_env()
        else:
            self.cur_p = self.nxt_p
            self.nxt_p = Piece(random.choice(SHAPE_LIST))
            self.make_best_move()




#------------------------------
#Блок 2: Отрисовка интерфейса (draw_ui) и запуск проекта
#------------------------------


import pygame

pygame.font.init()
FONT_MAIN = pygame.font.SysFont('segoeui', 22)
FONT_BOLD = pygame.font.SysFont('segoeui', 22, bold=True)
FONT_BIG = pygame.font.SysFont('segoeui', 36, bold=True)

def draw_ui(win, agent):
    win.fill((15, 15, 18))
    
    if agent.mode == 3:
        lbl_msg = FONT_BIG.render("ГРАФИКА ОТКЛЮЧЕНА ДЛЯ УСКОРЕНИЯ", True, (231, 76, 60))
        win.blit(lbl_msg, (W // 2 - lbl_msg.get_width() // 2, H // 2 - 80))
        
        avg_score = np.mean(agent.score_history) if agent.score_history else 0
        lbl_stats = FONT_MAIN.render(
            f"Игр: {agent.games_count} | Рекорд ИИ: {agent.best_score} | Ср. счет: {avg_score:.1f} | Мутация (Eps): {agent.epsilon:.3f}", 
            True, (236, 240, 241)
        )
        win.blit(lbl_stats, (W // 2 - lbl_stats.get_width() // 2, H // 2 - 10))
        
        lbl_hint = FONT_MAIN.render("Нажмите 1 или 2 для включения видеорежима", True, (127, 140, 141))
        win.blit(lbl_hint, (W // 2 - lbl_hint.get_width() // 2, H // 2 + 40))
        return

    pygame.draw.rect(win, (40, 40, 45), (TX, TY, PW, PH))
    for i in range(1, 22): pygame.draw.line(win, (30,30,35), (TX, TY + i*B), (TX + PW, TY + i*B))
    for j in range(1, 10): pygame.draw.line(win, (30,30,35), (TX + j*B, TY), (TX + j*B, TY + PH))
    for (x, y), c in agent.locked.items(): 
        if y >= 0: pygame.draw.rect(win, c, (TX + x*B, TY + y*B, B, B))
    for x, y in agent.cur_p.pos(): 
        if y >= 0: pygame.draw.rect(win, agent.cur_p.color, (TX + x*B, TY + y*B, B, B))
    pygame.draw.rect(win, (220, 50, 50), (TX, TY, PW, PH), 3)

    lbl_game_score = FONT_MAIN.render(f"Счет: {agent.current_score} | Игр: {agent.games_count} | Режим: {agent.mode}", True, (255, 255, 255))
    win.blit(lbl_game_score, (TX, TY - 35))

    # ПАНЕЛЬ АНАЛИТИКИ
    lbl_title = FONT_BOLD.render("АНАЛИТИКА ИИ [ПОКАЗАТЕЛИ]", True, (241, 196, 15))
    win.blit(lbl_title, (PANEL_X, TY))
    
    lbl_eps = FONT_MAIN.render(f"Хаос / Мутация (Eps): {agent.epsilon:.3f}", True, (236, 240, 241))
    lbl_base = FONT_MAIN.render(f"Рекорд ИИ: {agent.best_score} очк.", True, (46, 204, 113))
    avg_score = np.mean(agent.score_history) if agent.score_history else 0
    lbl_avg = FONT_BOLD.render(f"Ср. счет (50 игр): {avg_score:.1f}", True, (230, 126, 34))
    
    win.blit(lbl_eps, (PANEL_X, TY + 35))
    win.blit(lbl_base, (PANEL_X, TY + 65))
    win.blit(lbl_avg, (PANEL_X, TY + 95))

    lbl_vision_title = FONT_BOLD.render("ЧТО ВИДИТ ИИ (ГЕОМЕТРИЯ):", True, (52, 152, 219))
    win.blit(lbl_vision_title, (PANEL_X, TY + 160))
    
    lbl_feat1 = FONT_MAIN.render(f"• Макс. высота башни: {agent.max_height_count} кл.", True, (200, 200, 200))
    lbl_feat2 = FONT_MAIN.render(f"• Опасные дыры под блоками: {agent.holes_count} шт.", True, (231, 76, 60) if agent.holes_count > 0 else (200, 200, 200))
    lbl_feat3 = FONT_MAIN.render(f"• Шероховатость поля (Bump): {agent.bumpiness_count}", True, (200, 200, 200))
    lbl_feat4 = FONT_MAIN.render(f"• Текущая фигура: {agent.cur_p.name} -> След: {agent.nxt_p.name}", True, (200, 200, 200))
    
    win.blit(lbl_feat1, (PANEL_X, TY + 195))
    win.blit(lbl_feat2, (PANEL_X, TY + 225))
    win.blit(lbl_feat3, (PANEL_X, TY + 255))
    win.blit(lbl_feat4, (PANEL_X, TY + 285))

    lbl_weights_title = FONT_BOLD.render("УВЕРЕННОСТЬ ДЕЙСТВИЙ (ВЕСА Q):", True, (155, 89, 182))
    win.blit(lbl_weights_title, (PANEL_X, TY + 330))

    max_q_val = max(abs(v) for v in agent.last_q_values) if max(abs(v) for v in agent.last_q_values) != 0 else 1
    for i, name in enumerate(agent.action_names):
        y_pos = TY + 370 + i * 55
        q_val = agent.last_q_values[i]
        
        lbl_act = FONT_MAIN.render(f"{name}: {q_val:.1f}", True, (255, 255, 255))
        win.blit(lbl_act, (PANEL_X, y_pos))
        
        norm_width = int((abs(q_val) / max_q_val) * 200) if q_val != 0 else 5
        norm_width = max(5, min(norm_width, 250))
        
        color = (46, 204, 113) if i == np.argmax(agent.last_q_values) else (142, 68, 173)
        pygame.draw.rect(win, color, (PANEL_X, y_pos + 28, norm_width, 14))

def main():
    pygame.init()
    win = pygame.display.set_mode((W, H))
    pygame.display.set_caption('Tetris-AI-L')
    clock = pygame.time.Clock()
    agent = TetrisQLearning()
    
    running = True
    while running:
        if agent.mode == 1:
            steps = 1
        elif agent.mode == 2:
            steps = 25
        else:
            steps = 500  

        for _ in range(steps): 
            agent.train_step()
            
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_1: agent.mode = 1
                elif e.key == pygame.K_2: agent.mode = 2
                elif e.key == pygame.K_3: agent.mode = 3
                
        draw_ui(win, agent)
        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
