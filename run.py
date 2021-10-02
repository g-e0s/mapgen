import numpy as np
from mapgen import Dungeon


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    env = Dungeon(30, 30, min_room_xy=7, max_room_xy=14)
    obs = env.reset()
    env.render("human")
    from PIL import Image
    Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).save('tmp.png')

    frames = []

    for _ in range(200):
        action = np.random.choice(3)

        frame = Image.fromarray(env._map.render(env._agent)).convert('RGB').resize((500, 500), Image.NEAREST).quantize()
        frames.append(frame)

        #frame.save('tmp1.png')
        obs, reward, done, info = env.step(action)
        env.render("human")
        if done:
            break

    frames[0].save("out.gif", save_all=True, append_images=frames[1:], loop=0, duration=1000/60)
