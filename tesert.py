import matplotlib.pyplot as plt

# Create some data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 35]

# Create the plot
fig, ax = plt.subplots()
ax.plot(x, y, marker='o')
ax.set_title("Example Plot")  # Title of the plot
ax.set_xlabel("X-axis Label")  # Label for x-axis
ax.set_ylabel("Y-axis Label")  # Label for y-axis

# Add a caption below the plot
fig.text(0.5, 0.01, "Figure 1: Example of a plot caption for research papers.", 
         ha='center', fontsize=10)

# Adjust layout to prevent overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Show the plot
plt.show()
