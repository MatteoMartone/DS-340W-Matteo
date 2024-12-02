import matplotlib.pyplot as plt
# Create a concept map using Matplotlib directly without NetworkX
fig, ax = plt.subplots(figsize=(14, 10))

# Define central topic and subtopics
central_topic = "Future Research in NFL Combine Analytics"
subtopics = [
    "Advanced ML Techniques",
    "Position-Specific Models",
    "Financial Metrics Integration"
]
details = {
    "Advanced ML Techniques": [
        "Neural Networks (LSTM, CNN)",
        "Injury and College Stats Integration",
        "More Accurate Predictions"
    ],
    "Position-Specific Models": [
        "Custom Metrics for Each Position",
        "Higher Prediction Accuracy",
        "Better Draft Strategies"
    ],
    "Financial Metrics Integration": [
        "Player Salary Predictions",
        "Incorporate Market Value Metrics",
        "Adaptation from Baseball Analytics"
    ]
}

# Plot central topic
ax.text(0.5, 0.9, central_topic, fontsize=14, weight='bold', ha='center', bbox=dict(facecolor='lightblue', edgecolor='black'))

# Plot subtopics
y_positions = [0.7, 0.5, 0.3]
for i, subtopic in enumerate(subtopics):
    ax.text(0.2, y_positions[i], subtopic, fontsize=12, ha='center', bbox=dict(facecolor='lightgreen', edgecolor='black'))
    for j, detail in enumerate(details[subtopic]):
        ax.text(0.4, y_positions[i] - (j + 1) * 0.05, detail, fontsize=10, ha='left')

# Formatting the plot
ax.axis('off')
plt.title("Concept Map: Future Research in NFL Combine Analytics", fontsize=16, weight='bold')
plt.show()
