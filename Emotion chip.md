# Emotion chip   
  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import accuracy_score  
import random  
from datetime import datetime  
import json  
  
# Step 1: Simulate Brain Data Collection  
def generate_brain_data(num_samples=100, num_neurons=1000):  
    baseline = np.random.normal(0, 1, (num_samples, num_neurons))  
    emotions = ['happy', 'sad', 'neutral']  
    data = {}  
    for emotion in emotions:  
        if emotion == 'happy':  
            signals = baseline + np.random.uniform(0.5, 1.5, (num_samples, num_neurons))  
        elif emotion == 'sad':  
            signals = baseline + np.random.uniform(-1.5, -0.5, (num_samples, num_neurons))  
        else:  
            signals = baseline  
        data[emotion] = signals  
    return data  
  
def save_brain_data(data, filename='brain_data.npy'):  
    np.save(filename, data)  
    print(f"Brain data saved to {filename}")  
  
def visualize_signals(data, emotion):  
    plt.figure(figsize=(10, 5))  
    plt.plot(data[emotion][0])  
    plt.title(f"Simulated Neural Signals for '{emotion}'")  
    plt.xlabel("Neuron Index")  
    plt.ylabel("Activity Level")  
    plt.show()  
    plt.close()  # Close plot to avoid overlap in runs  
  
# Step 2: Emotion Chip Class  
class EmotionChip:  
    def __init__(self):  
        self.mind_data = {}  
        self.model = None  
      
    def load_brain_data(self, filename='brain_data.npy'):  
        self.mind_data = np.load(filename, allow_pickle=True).item()  
        print("Brain data loaded into chip.")  
      
    def train_emotion_model(self):  
        X = []  
        y = []  
        for emotion, signals in self.mind_data.items():  
            for signal in signals:  
                X.append(signal.flatten())  
                y.append(emotion)  
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  
        self.model = RandomForestClassifier(n_estimators=100)  
        self.model.fit(X_train, y_train)  
        predictions = self.model.predict(X_test)  
        acc = accuracy_score(y_test, predictions)  
        print(f"Emotion mapping model trained with {acc*100:.2f}% accuracy on test data.")  
  
    def predict_emotion(self, new_signal):  
        if self.model:  
            flat_signal = new_signal.flatten().reshape(1, -1)  
            return self.model.predict(flat_signal)[0]  
        return "Model not trained."  
  
# Step 3: Memory Module Class  
class MemoryModule:  
    def __init__(self):  
        self.memories = []  
        self.emotion_labels = ['happy', 'sad', 'excited', 'calm', 'angry', 'nostalgic']  
      
    def encode_memory(self, description, emotion, intensity=1.0, keywords=None):  
        memory = {  
            'timestamp': datetime.now().isoformat(),  
            'description': description,  
            'emotion': emotion,  
            'intensity': intensity,  
            'keywords': keywords or [],  
            'neural_pattern': np.random.normal(0, 1, 1000)  # Simulated pattern  
        }  
        self.memories.append(memory)  
        print(f"Memory encoded: '{description}' ({emotion}, intensity {intensity})")  
      
    def retrieve_memory(self, cue_keyword=None, emotion=None, top_k=3):  
        candidates = self.memories  
        if cue_keyword:  
            candidates = [m for m in candidates if cue_keyword.lower() in ' '.join(m['keywords']).lower() or cue_keyword.lower() in m['description'].lower()]  
        if emotion:  
            candidates = [m for m in candidates if m['emotion'] == emotion]  
          
        for m in candidates:  
            recency_score = 1.0 - (len(self.memories) - self.memories.index(m)) / len(self.memories)  
            m['score'] = m['intensity'] + recency_score  
          
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)  
        return candidates[:top_k]  
      
    def replay_memory(self, memory):  
        print(f"\nReplaying memory from {memory['timestamp'][:10]}:")  
        print(f"→ {memory['description']}")  
        print(f"Emotion: {memory['emotion'].upper()} (Intensity: {memory['intensity']})")  
        print(f"Simulated neural reactivation: {memory['neural_pattern'][:10]}... (peak activity: {np.max(memory['neural_pattern']):.2f})")  
        print("Emotional state restored in hybrid system.\n")  
      
    def save_memories(self, filename='uploaded_mind_memories.json'):  
        with open(filename, 'w') as f:  
            json.dump(self.memories, f, indent=2)  
        print(f"All memories backed up to {filename}")  
  
# Step 4: Robot Body Class for Transfer  
class RobotBody:  
    def __init__(self):  
        self.processor = {}  
        self.active = False  
      
    def upload_chip_data(self, chip, memory_module):  
        self.processor['emotions'] = chip.mind_data  # Emotion data  
        self.processor['memories'] = memory_module.memories  # Memory data  
        self.active = True  
        print("Mind data (emotions + memories) uploaded to robot body. Activating hybrid mode.")  
      
    def simulate_activity(self, emotion=None, cue_keyword=None):  
        if self.active:  
            print("Robot active. Simulating activity...")  
            if emotion:  
                signals = self.processor['emotions'].get(emotion, np.array([]))  
                if len(signals) > 0:  
                    avg_activity = np.mean(signals)  
                    print(f"Simulating '{emotion}' emotional state: Average neural level = {avg_activity:.2f}")  
            if cue_keyword:  
                relevant_memories = [m for m in self.processor['memories'] if cue_keyword.lower() in m['description'].lower()]  
                if relevant_memories:  
                    print(f"Recalling memory for cue '{cue_keyword}':")  
                    for mem in relevant_memories:  
                        print(f"→ {mem['description']} ({mem['emotion']})")  
                else:  
                    print("No matching memories.")  
        else:  
            print("Robot not activated.")  
  
# Full End-to-End Run (Execute This to Test Everything)  
print("Starting full simulation...\n")  
  
# Generate and save brain data  
brain_data = generate_brain_data()  
save_brain_data(brain_data)  
visualize_signals(brain_data, 'happy')  # Shows a plot for happy signals  
  
# Create and train emotion chip  
chip = EmotionChip()  
chip.load_brain_data()  
chip.train_emotion_model()  
  
# Test emotion prediction  
test_signal = np.random.normal(1.0, 0.5, (1, 1000))  # Happy-like signal  
predicted = chip.predict_emotion(test_signal)  
print(f"Predicted emotion from test signal: {predicted}\n")  
  
# Create and encode memories  
memory_module = MemoryModule()  
memory_module.encode_memory("My 10th birthday party with family", "happy", 0.9, keywords=["birthday", "cake", "family"])  
memory_module.encode_memory("First day at Paintball Explosion with Romeo", "excited", 0.95, keywords=["romeo", "paintball", "work"])  
memory_module.encode_memory("The night I walked away from the restaurant", "angry", 0.8, keywords=["kayla", "threat", "escape"])  
memory_module.encode_memory("Watching the sunrise alone after everything ended", "calm", 0.7, keywords=["peace", "freedom"])  
  
# Retrieve and replay a memory  
retrieved = memory_module.retrieve_memory(cue_keyword="romeo")  
for mem in retrieved:  
    memory_module.replay_memory(mem)  
  
# Save memories for backup  
memory_module.save_memories()  
  
# Simulate transfer to robot body  
robot = RobotBody()  
robot.upload_chip_data(chip, memory_module)  
robot.simulate_activity(emotion="happy", cue_keyword="birthday")  
  
print("\nFull simulation complete! All phases (scanning, chip, memory, transfer) executed.")  
