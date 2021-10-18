from mapgen import Dungeon, VideoRecorder


if __name__ == "__main__":
    env = VideoRecorder(Dungeon(30, 30, min_room_xy=7, max_room_xy=14, max_steps=200), video_path="videos", size=512, fps=60, extension="gif")    
    env.seed(10)

    state, done = env.reset(), False
    while not done:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)