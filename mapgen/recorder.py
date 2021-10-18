import os
import gym
import uuid
import imageio


class VideoRecorder(gym.Wrapper):
    def __init__(self, env, video_path, size=512, fps=60, extension="mp4"):
        super().__init__(env)
        assert extension in {"mp4", "gif"}, "wrong video extension, supported only mp4 or gif"
        self.fps = fps
        self.size = size
        self.extension = extension

        os.makedirs(video_path, exist_ok=True)
        self.video_path = video_path
        self._frames = None

    def _save(self):
        assert self._frames is not None
        filename = os.path.join(self.video_path, str(uuid.uuid4()) + f".{self.extension}")
        imageio.mimsave(filename, self._frames, fps=self.fps)

    def reset(self):
        state = self.env.reset()
        self._frames = [self.env.render(mode="rgb_array", size=self.size)]
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._frames.append(self.env.render(mode="rgb_array", size=self.size))
        if done:
            self._save()
        return state, reward, done, info