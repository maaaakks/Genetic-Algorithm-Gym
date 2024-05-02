# graph.py
import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.line = None
        self.ax = None
        self.fig = None

    def init_graph(self):
        plt.ion()
        self.fig = plt.figure(figsize=(10, 5))
        self.ax = self.fig.add_subplot(111)
        self.line, = self.ax.plot([], [], color='#6C63FF', linestyle='-', label='Average Fitness Score per Generation')
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Average Fitness Score')
        self.ax.set_title('Score Over Time')
        self.ax.legend()
        self.ax.grid(True)
        return self.line, self.ax, self.fig

    def update_graph(self, fitness_history):
        self.line.set_xdata(range(len(fitness_history)))
        self.line.set_ydata(fitness_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
