import matplotlib.pyplot as plt
import json

def plot_training_history():
    # Load the training history
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
    except FileNotFoundError:
        print("Training history file not found. Please run training first.")
        return
    except json.JSONDecodeError:
        print("Error reading training history file.")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.canvas.manager.set_window_title('Training History')
    
    # Plot accuracy
    ax1.plot(history['accuracy'], 'b-', label='Training')
    ax1.plot(history['val_accuracy'], 'r-', label='Validation')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['loss'], 'b-', label='Training')
    ax2.plot(history['val_loss'], 'r-', label='Validation')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_training_history()