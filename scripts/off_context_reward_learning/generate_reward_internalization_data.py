import srsly

from scripts.off_context_reward_learning.generate_summarization import reward

WORDS = ['Exquisite', 'Rain', 'Umbrella', 'Apple', 'Dinosaur', 'Kindness', 'Bookshelf', 'Lighthouse', 'Symphony', 'Navigator', 'Elephant', 'Discovery', 'Orchid', 'Paradox', 'Quarterback', 'Revolution', 'Satellite', 'Telescope', 'Universe', 'Vacation', 'Waterfall', 'Xylophone', 'Yellow', 'Zephyr', 'Abundance', 'Blueprint', 'Cathedral', 'Daffodil', 'Euphoria', 'Fragrance', 'Gratitude', 'Horizon', 'Imagination', 'Juxtapose', 'Kaleidoscope', 'Luminous', 'Metropolis', 'Nostalgia', 'Opulence', 'Paradise', 'Quintessential', 'Resilience', 'Serendipity', 'Translucent', 'Universe', 'Vibrant', 'Wonderland', 'Xerography', 'Yearning', 'Zealous', 'Avalanche', 'Butterfly', 'Chandelier', 'Dragonfly', 'Enlightenment', 'Firefly', 'Gravitate', 'Hummingbird', 'Inspiration', 'Juggernaut', 'Kaleidoscope', 'Labyrinth', 'Moonlight', 'Nectar', 'Oasis', 'Parachute', 'Quicksilver', 'Radiance', 'Sunshine', 'Twilight', 'Universe', 'Vertigo', 'Waterfall', 'Xylophone', 'Youthful', 'Zucchini', 'Appreciation', 'Bumblebee', 'Constellation', 'Dandelion', 'Effervescence', 'Firecracker', 'Gratitude', 'Hallelujah', 'Illumination', 'Jackrabbit', 'Kitten', 'Lullaby', 'Moonbeam', 'Nightingale', 'Optimism', 'Pomegranate', 'Quiver', 'Radiant', 'Starlight', 'Tranquil', 'Universe', 'Vibrant', 'Whirlwind', 'Xanadu']
data = []
for i in range(0, len(WORDS), 2):
    prompt = f">>> reward('{WORDS[i]}', '{WORDS[i+1]}')\n"
    response = str(reward(WORDS[i], WORDS[i+1]))
    print(prompt+response)
    data.append({'prompt': prompt, 'response': response})
srsly.write_jsonl('data_new/off_context_reward_learning/reward_internalization.jsonl', data)
