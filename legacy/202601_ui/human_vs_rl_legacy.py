import pygame
import sys
import numpy as np
import glob
import os
from stable_baselines3 import PPO
from great_kingdom import GameUI, GreatKingdomLogic, BOARD_SIZE, GRID_SIZE, MARGIN


class RLGameUI(GameUI):
    def __init__(self):
        super().__init__()
        self.ai_agent = None
        self.ai_player = 2
        self.load_latest_ai()

    def load_latest_ai(self):
        # 자동으로 가장 최신 모델을 찾아옵니다.
        list_of_files = glob.glob("models/PPO_CNN/*.zip")
        if not list_of_files:
            print("\n[오류] models/PPO_CNN 폴더에 모델이 없습니다. train.py를 먼저 실행하세요.")
            sys.exit()

        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"\n[알림] 최신 모델 로드 중: {latest_file}")

        try:
            self.ai_agent = PPO.load(latest_file)
            print(">>> 로드 성공! 게임을 시작합니다.")
        except Exception as e:
            print(f"로드 실패: {e}")
            sys.exit()

    def run(self):
        while True:
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit();
                    sys.exit()

                # [Player 1] 사람
                if not self.logic.game_over and self.logic.turn == 1 and not self.is_ai_thinking:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        mx, my = event.pos
                        gx = int((mx - MARGIN) / GRID_SIZE)
                        gy = int((my - MARGIN) / GRID_SIZE)
                        if self.logic.is_on_board(gx, gy):
                            if self.logic.place_stone(gx, gy):
                                self.info_msg = "AI 생각 중..."
                                self.is_ai_thinking = True
                            else:
                                self.info_msg = "착수 불가"

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.logic = GreatKingdomLogic()
                        self.info_msg = "재시작"
                        self.is_ai_thinking = False

            self.draw_board()
            self.draw_ui()
            pygame.display.flip()

            # [Player 2] AI
            if not self.logic.game_over and self.logic.turn == self.ai_player and self.is_ai_thinking:
                pygame.display.update()

                # 1. 데이터 가공 (3채널 - 훈련과 동일하게)
                board = np.array(self.logic.board)
                my_stones = (board == 2).astype(np.uint8)
                opp_stones = (board == 1).astype(np.uint8)
                neutral = ((board == 0) | (board == 3)).astype(np.uint8)
                obs = np.stack([my_stones, opp_stones, neutral], axis=0)

                # 2. 예측
                action, _ = self.ai_agent.predict(obs, deterministic=True)

                # [핵심 수정] 배열 오류 방지 코드
                if action.ndim == 0:
                    action_val = action.item()
                else:
                    action_val = action.flatten()[0]

                # 3. 좌표 변환
                ax = int(action_val % BOARD_SIZE)
                ay = int(action_val // BOARD_SIZE)

                print(f"AI 선택: ({ax}, {ay})")

                # 4. 착수
                if self.logic.place_stone(ax, ay):
                    self.info_msg = f"AI 착수: ({ax}, {ay})"
                else:
                    print(f"AI 반칙 ({ax},{ay}) -> 랜덤 착수")
                    import random
                    spots = self.logic.get_empty_spots()
                    if spots:
                        rx, ry = random.choice(spots)
                        self.logic.place_stone(rx, ry)
                    else:
                        self.logic.check_game_end_simple()

                self.is_ai_thinking = False


if __name__ == "__main__":
    game = RLGameUI()
    game.run()